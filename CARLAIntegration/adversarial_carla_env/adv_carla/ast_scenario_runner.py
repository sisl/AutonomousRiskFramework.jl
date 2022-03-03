import carla
import traceback
import os
import time
import sys
import py_trees

if 'SCENARIO_RUNNER_ROOT' not in os.environ:
    raise Exception("Please add 'SCENARIO_RUNNER_ROOT' to your environment variables.")
else:
    sys.path.append(os.getenv('SCENARIO_RUNNER_ROOT')) # Add scenario_runner package to import path

from scenario_runner import ScenarioRunner
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from pdb import set_trace as breakpoint # DEBUG. TODO!

class ASTScenarioRunner(ScenarioRunner):

    def __init__(self, args, *, remove_other_actors=True):
        super().__init__(args)
        self._args.reloadWorld = False # Force no-reload.
        self._args.record = False
        self.load_count = 0
        self.start_time = None
        self.recorder_name = None
        self.scenario = None
        self.running = False
        self.remove_other_actors = remove_other_actors


    def _load_and_run_scenario(self, config):
        self.finished = False
        self.scenario_config = config


    def load_scenario(self):
        """
        Load and run the scenario given by config
        """
        self.finished = False

        if self.running:
            result = self._stop_scenario(self.start_time, self.recorder_name, self.scenario)
            self._cleanup()

        self.scenario_config.name = "RouteScenario_" + str(self.load_count)

        if not self._load_and_wait_for_world(self.scenario_config.town, self.scenario_config.ego_vehicles):
            self._cleanup()
            return None, None, None # load failed.

        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                self.scenario_config.agent = self.agent_instance
            except Exception as e:          # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return None, None, None # load failed.

        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))
        tm = self.client.get_trafficmanager(int(self._args.trafficManagerPort))
        tm.set_random_device_seed(int(self._args.trafficManagerSeed))
        if self._args.sync:
            tm.set_synchronous_mode(False)

        # Prepare scenario
        print("Preparing scenario: " + self.scenario_config.name)

        if self.remove_other_actors:
            RouteScenario._initialize_actors = _initialize_actors_no_background # Monkey patching to avoid background actors
        else:
            RouteScenario._initialize_actors = _initialize_actors_reduced_background # Monkey patching to reduce background actors

        try:
            self._prepare_ego_vehicles(self.scenario_config.ego_vehicles)
            if self._args.openscenario:
                self.scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=self.scenario_config,
                                        config_file=self._args.openscenario,
                                        timeout=100000)
            elif self._args.route:
                self.scenario = RouteScenario(world=self.world,
                                         config=self.scenario_config,
                                         debug_mode=self._args.debug)
            else:
                scenario_class = self._get_scenario_class_or_fail(self.scenario_config.type)
                self.scenario = scenario_class(self.world,
                                          self.ego_vehicles,
                                          self.scenario_config,
                                          self._args.randomize,
                                          self._args.debug)
        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return None, None, None # load failed.

        # Change max trigger distance between route and scenario # TODO: Needed?
        # self.scenario.behavior.children[0].children[0].children[0]._distance = 5.5 # was 1.5 in route_scenario.py: _create_behavior()

        try:
            if self._args.record:
                self.recorder_name = "{}/{}/{}.log".format(
                    os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record, config.name)
                print(self.recorder_name)
                self.client.start_recorder(self.recorder_name, True)

            # Load scenario and run it
            self.manager.load_scenario(self.scenario, self.agent_instance)

            self.manager.start_system_time = time.time()
            self.start_time = GameTime.get_time()
            self.manager._watchdog = Watchdog(float(self._args.timeout))
            self.manager._watchdog.start()
            self.manager._running = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self.load_count += 1
        self.test_status = "INIT"
        self.running = True

        return self.start_time, self.recorder_name, self.scenario


    def _check_failures(self):
        failure = False
        for i, criterion in enumerate(self.manager.scenario.get_criteria()):
            if i!=1:
                continue
            if (not criterion.optional and
                    criterion.test_status == "FAILURE" and
                        self.test_status != "FAILURE"):
                failure = True
            elif criterion.test_status == "ACCEPTABLE":
                failure = False
        if failure:
            self.test_status = "FAILURE"

        return failure


    def _stop_scenario(self, start_time, recorder_name, scenario):
        try:
            self._clean_scenario_ast(start_time)

            # Identify which criteria were met/not met
            # self._analyze_scenario(config)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(self.manager.scenario.get_criteria(), recorder_name)
            result = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self.finished = False
        self._cleanup()
        return result


    def _clean_scenario_ast(self, start_game_time):
        # self.manager._watchdog.stop()
        self.manager.cleanup()

        self.manager.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.manager.scenario_duration_system = self.manager.end_system_time - \
            self.manager.start_system_time
        self.manager.scenario_duration_game = end_game_time - start_game_time

        if self.manager.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("(AdversarialCARLAEnv) ScenarioManager: Terminated due to failure")


    def parse_scenario(self):
        self.finished = False
        super().run()


def _initialize_actors_no_background(self, config):
    """
    Set other_actors to the superset of all scenario actors
    NOTE: monkey patching _initialize_actors from route_scenario.py
    """

    # Add all the actors of the specific scenarios to self.other_actors
    for scenario in self.list_scenarios:
        self.other_actors.extend(scenario.other_actors)


def _initialize_actors_reduced_background(self, config):
    # Create the background activity of the route
    # NOTE: Decreased these by an order of magnitude
    town_amount = {
        'Town01': 12,
        'Town02': 10,
        'Town03': 12,
        'Town04': 20,
        'Town05': 12,
        'Town06': 15,
        'Town07': 11,
        'Town08': 18,
        'Town09': 30,
        'Town10': 12,
    }

    amount = town_amount[config.town] if config.town in town_amount else 0

    new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                            amount,
                                                            carla.Transform(),
                                                            autopilot=True,
                                                            random_location=True,
                                                            rolename='background')

    if new_actors is None:
        raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

    for _actor in new_actors:
        self.other_actors.append(_actor)

    _initialize_actors_no_background(self, config)