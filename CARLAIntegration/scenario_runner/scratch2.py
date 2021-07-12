#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA scenario_runner

This is the main script to be executed when running a scenario.
It loads the scenario configuration, loads the scenario and manager,
and finally triggers the scenario execution.
"""

from __future__ import print_function

import glob
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import os
import signal
import sys
import time
import json
# import pkg_resources
# import numpy as np
from julia import Main, Pkg
Pkg.activate(r'Project.toml')
Main.include("ast_tools.jl")
Main.include("visualization.jl")

import carla

from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# Version of scenario_runner
VERSION = '0.9.11'