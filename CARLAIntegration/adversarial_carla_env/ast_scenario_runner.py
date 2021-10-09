import traceback
import os
import time
import sys

sys.path.append("../scenario_runner") # Add scenario_runner package to import path

from scenario_runner import ScenarioRunner
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.scenariomanager.timer import GameTime

from pdb import set_trace as breakpoint # DEBUG. TODO!

class ASTScenarioRunner(ScenarioRunner):

    def __init__(self, args):
        super().__init__(args)
        self._args.reloadWorld = False # Force no-reload.
        self._args.record = False
        self.load_count = 0
        self.start_time = None
        self.recorder_name = None
        self.scenario = None
        self.first_load = True


    def _load_and_run_scenario(self, config):
        self.scenario_config = config


    def load_scenario(self):
        """
        Load and run the scenario given by config
        """
        if not self.first_load:
            result = self._stop_scenario(self.start_time, self.recorder_name, self.scenario)
            self._cleanup()
            self.first_load = False

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

        # Prepare scenario
        print("Preparing scenario: " + self.scenario_config.name)

        RouteScenario._initialize_actors = _initialize_actors # Monkey patching to avoid background actors

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
            self.manager._watchdog.start()
            self.manager._running = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self.load_count += 1
        self.test_status = "INIT"

        return self.start_time, self.recorder_name, self.scenario


    def parse_scenario(self):
        super().run()


def _initialize_actors(self, config):
    """
    Set other_actors to the superset of all scenario actors
    NOTE: monkey patching _initialize_actors from route_scenario.py
    """

    # Add all the actors of the specific scenarios to self.other_actors
    for scenario in self.list_scenarios:
        self.other_actors.extend(scenario.other_actors)
