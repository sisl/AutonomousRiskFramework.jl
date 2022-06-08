import xml.etree.ElementTree as ET
from io import BytesIO
from lxml import etree
import json
import numpy as np
from tqdm import tqdm
import shutil
import os


def abspath(filename):
   dirname = os.path.dirname(__file__)
   return os.path.join(dirname, filename)

def get_local_rng(seed):
   # Use local RNG so we do not set the global seed
   print("Generating scenario with seed:", seed)
   return np.random.default_rng(seed)

def read_routes(routes_xml_file_name):
   tree = ET.parse(routes_xml_file_name)
   routes_list = [r for r in tree.iter("route")]
   waypoint_list_route = [[w.attrib for w in r.iter("waypoint")] for r in routes_list]
   # convert waypointlist into np arrays
   routes_array_list = []
   for i in range(len(waypoint_list_route)):
      head = routes_list[i].attrib
      no_waypoints = len(waypoint_list_route[i])
      array = np.zeros((no_waypoints,6))
      route = waypoint_list_route[i]
      array[:,0] = [w['x'] for w in route]
      array[:,1] = [w['y'] for w in route]
      array[:,2] = [w['z'] for w in route]
      array[:,3] = [w['pitch'] for w in route]
      array[:,4] = [w['roll'] for w in route]
      array[:,5] = [w['yaw'] for w in route]
      head["waypoints"] = array
      routes_array_list.append(head)
   del tree
   del routes_list
   del waypoint_list_route
   return routes_array_list

def read_scenarios(scenarios_json_file_name):
   with open(scenarios_json_file_name) as f:
      obj = json.load(f)['available_scenarios'][0]

   scenario_list = []
   raw_list = []
   counter = 0
   for town in list(obj.keys()):
      obj_t = obj[town]
      for s in obj_t:
         scenario = s["scenario_type"]
         for c in s["available_event_configurations"]:
            scenario_dict = {}
            scenario_dict["id"] = counter
            counter += 1
            scenario_dict["town"] = town
            scenario_dict["scenario_type"] = scenario
            scenario_dict["waypoint"] = np.array([float(c["transform"]["x"]),float(c["transform"]["y"]),float(c["transform"]["z"]),float(c["transform"]["pitch"]),0,float(c["transform"]["yaw"])])
            scenario_list.append(scenario_dict)
            raw_list.append(c)

   return scenario_list, obj, raw_list

def distance_point(x1,y1,x2,y2,x,y,error_bound):
   def intersection_point(x1,y1,x2,y2,x3,y3,x4,y4):
      # for x coordinate
      a1 = np.array([[x1,y1],[x2,y2]])
      a2 = np.array([[x3,y3],[x4,y4]])
      A = np.array([[np.linalg.det(a1),x1-x2],[np.linalg.det(a2),x3-x4]])
      B = np.array([[x1-x2,y1-y2],[x3-x4,y3-y4]])
      x = np.linalg.det(A)/np.linalg.det(B)

      # for y coordinate
      C = np.array([[np.linalg.det(a1),y1-y2],[np.linalg.det(a2),y3-y4]])
      y = np.linalg.det(C)/np.linalg.det(B)

      return x,y
   
   def distance_two_points(x1,y1,x2,y2):
      return np.sqrt((x2-x1)**2+(y2-y1)**2)

   # distance of point to line
   d = np.abs((x2-x1)*(y1-y)-(x1-x)*(y2-y1))/np.sqrt((x2-x1)**2+(y2-y1)**2)
   
   # distance from points 1 and 2
   dp1 = distance_two_points(x1,y1,x,y)
   dp2 = distance_two_points(x2,y2,x,y)

   # normal vector line
   R = np.array([[0,-1],[1,0]]) #rotation matrix
   t = np.array([[x2-x1],[y2-y1]])
   n = R @ t

   # find closest point on line
   xi, yi = intersection_point(x1,y1,x2,y2,x,y,x+n[0].item(),y+n[1].item())

   # check if intersection point is between points
   e = distance_two_points(x1,y1,xi,yi) + distance_two_points(xi,yi,x2,y2) - distance_two_points(x1,y1,x2,y2)
   if (e > 2*error_bound) or (d>error_bound):
      flag = False
   else:
      flag = True
   if e > 0.0001:
      d = min(dp1,dp2)

   return d, flag

def find_route_snippet(rng, scenario_dict, routes_list, error_bound, randomize_first_point=True, *, unavailable_towns=["Town06"]):
   """
   randomize_first_point: randomly initialize the first point between the points on
   """
   town = scenario_dict["town"]
   scenario_type = scenario_dict["scenario_type"]
   x = scenario_dict["waypoint"][0]
   y = scenario_dict["waypoint"][1]
   
   candidates = []
   for route in routes_list:
      town_route = route["town"]
      if town in unavailable_towns:
         continue
      if town == town_route:
         no_waypoints = route["waypoints"].shape[0]
         wps = route["waypoints"]

         wp_hits = []
         min_distance = 1000000
         for i in range(no_waypoints-1):
            x1 = wps[i,0]
            y1 = wps[i,1]
            x2 = wps[i+1,0]
            y2 = wps[i+1,1]
            d,flag = distance_point(x1,y1,x2,y2,x,y,error_bound)
            if flag == True:
               wp_hits.append(i)
               if d<min_distance:
                  min_distance = d
         try:
            first_hit = min(wp_hits)
            if first_hit>0:
               first_route_wp = first_hit-1
            else:
               first_hit = 0
            if randomize_first_point:
               rand_fraction = rng.random()
               candidates.append({"id": route["id"],"town":town,"distance":min_distance,"first_wp":first_route_wp,"rand_fraction":rand_fraction, "trigger_wp":[x, y]})
            else:
               candidates.append({"id": route["id"],"town":town,"distance":min_distance,"first_wp":first_route_wp,"rand_fraction":0, "trigger_wp":[x, y]})
         except:
            continue
   if not candidates:   #no candidates found
      print("No routes found in the dataset")
   
   
   #find best candidate
   min_distance_routes = 1000000
   for c in candidates:
      if c["distance"]<min_distance_routes:
         candidate = c
   if len(candidates) == 0:  
      candidate = None
   return candidate




def write_route(rng, candidate, routes, weather="Random", fname=abspath("data/test.xml")):
   def write_xml(route, route_id, town, fname):
      def rand_number_range(min,max):
         return rng.random()*(max-min)+min

      no_waypoints = route.shape[0]
      root = ET.Element('routes')
      r = ET.SubElement(root,'route')
      r.set("id",route_id)
      r.set("town",town)

      #set weather randomly
      if weather == "Random":
         wx = ET.SubElement(r,'weather')
         wx.set("cloudiness",str(rand_number_range(0,100)))
         wx.set("precipitation",str(rand_number_range(0,100)))
         wx.set("precipitation_deposits",str(rand_number_range(0,100)))
         wx.set("wind_intensity",str(rand_number_range(0,100)))
         wx.set("sun_azimuth_angle",str(rand_number_range(0,360)))
         wx.set("sun_altitude_angle",str(rand_number_range(-20,90)))
         wx.set("fog_density",str(rand_number_range(0,100)))
         wx.set("fog_distance",str(rand_number_range(0,100)))
         wx.set("wetness",str(rand_number_range(0,100)))
         del wx
      else:
         # `whether` is a dictionary
         wx = ET.SubElement(r,'weather')
         wx.set("cloudiness",str(weather['cloudiness']))
         wx.set("precipitation",str(weather['precipitation']))
         wx.set("precipitation_deposits",str(weather['precipitation_deposits']))
         wx.set("wind_intensity",str(weather['wind_intensity']))
         wx.set("sun_azimuth_angle",str(weather['sun_azimuth_angle']))
         wx.set("sun_altitude_angle",str(weather['sun_altitude_angle']))
         wx.set("fog_density",str(weather['fog_density']))
         wx.set("fog_distance",str(weather['fog_distance']))
         wx.set("wetness",str(weather['wetness']))
         del wx

      # waypoints
      for i in range(no_waypoints):
         w = ET.SubElement(r,'waypoint')
         w.set("pitch",str(route[i,3]))
         w.set("roll",str(route[i,4]))
         w.set("x",str(route[i,0]))
         w.set("y",str(route[i,1]))
         w.set("yaw",str(route[i,5]))
         w.set("z",str(route[i,2]))
         del w
      et = ET.ElementTree("tree")
      et._setroot(root)
      et.write(fname, encoding = "UTF-8", xml_declaration = True)
      root.clear()
      r.clear()
      del root
      del et
      del r

      # #prettify
      # tree = etree.parse(fname)
      # pretty = etree.tostring(tree, encoding="UTF-8", pretty_print=True)
      # print(pretty)
      print("stop")


   route_id = candidate["id"]
   town = candidate["town"]
   first_wp = candidate["first_wp"]
   # included the randomization of the first waypoint

   route = routes[int(route_id)]["waypoints"][first_wp:,:]
   
   # Match the trigerring waypoint exactly
   # route[5, 0] = candidate["trigger_wp"][0]
   # route[5, 1] = candidate["trigger_wp"][1]
   
   print("Town: ", town)
   print("First waypoint: ", routes[int(route_id)]["waypoints"][first_wp, :])
   print("Second waypoint: ", routes[int(route_id)]["waypoints"][first_wp+1, :])
   spectator_loc = [routes[int(route_id)]["waypoints"][first_wp, 0], routes[int(route_id)]["waypoints"][first_wp, 1], 0.0]
   # print("Initialization command: \npython config.py --map {town} --port 2222 --spectator-loc {x} {y} {z}".format(town=town, x=spectator_loc[0], y=spectator_loc[1], z=spectator_loc[2]))
   write_xml(route,route_id,town,fname)
   return fname, town, spectator_loc


def write_scenario_json(scenario, obj, raw_list, fname=abspath("data/test.json")):
   scenario_id = scenario["id"]
   scenario_type = scenario["scenario_type"]
   other_actor_type = scenario["other_actor_type"]
   town = scenario["town"]
   json_dict = {}
   json_dict["available_scenarios"] = [{town:[{"available_event_configurations":[raw_list[scenario_id]],"scenario_type":scenario_type,"other_actor_type":other_actor_type}]}]
   with open(fname,"w") as f:
      json.dump(json_dict,f,indent=4)
   return fname

# NUMBER_CLASS_TRANSLATION = {
#     "Scenario1": ControlLoss,
#     "Scenario2": FollowLeadingVehicle,
#     "Scenario3": DynamicObjectCrossing,
#     "Scenario4": VehicleTurningRoute,
#     "Scenario5": OtherLeadingVehicle,
#     "Scenario6": ManeuverOppositeDirection,
#     "Scenario7": SignalJunctionCrossingRoute,
#     "Scenario8": SignalJunctionCrossingRoute,
#     "Scenario9": SignalJunctionCrossingRoute,
#     "Scenario10": NoSignalJunctionCrossingRoute
# }
def create_random_scenario(*, seed=None, scenario_type="Scenario4", other_actor_type="vehicle.audi.tt", weather="Random"):
   if seed is None:
      maxint32 = np.iinfo(np.int32).max
      seed = np.random.randint(low=0, high=maxint32)
   rng = get_local_rng(seed)
   routes = read_routes(abspath("data/routes_training.xml"))
   scenarios, obj, raw_list = read_scenarios(abspath("data/all_towns_traffic_scenarios.json"))
   no_scenarios = len(scenarios)
   
   candidate, counter  = None, 0
   # specify allowed types here
   # allowed_scenario_types = ["Scenario2", "Scenario3", "Scenario4", "Scenario5", "Scenario6", "Scenario7", "Scenario8", "Scenario9", "Scenario10"]
   # allowed_scenario_types = ["Scenario4"]
   allowed_scenario_types = [scenario_type]

   while candidate is None:
      # select random scenario
      scenario_type = None
      while scenario_type not in allowed_scenario_types:
        rand_scenario_id = rng.choice(range(no_scenarios))
        the_scenario = scenarios[rand_scenario_id]
        scenario_type = the_scenario["scenario_type"]
      
      print("Scenario: ", the_scenario)
      
      # find candidate route
      candidate = find_route_snippet(rng, the_scenario, routes, 1)
      print(counter)
      counter += 1
   route_filename, town, spectator_loc = write_route(rng, candidate, routes, weather=weather)
   the_scenario["other_actor_type"] = other_actor_type
   scenario_filename = write_scenario_json(the_scenario, obj, raw_list)

   # print("Run command:")
   # print(r"python scenario_runner_ast_gym.py --route data/test.xml data/test.json --port 2222 --agent srunner/autoagents/ast_agent.py --record recordings --timeout 60")
   return route_filename, scenario_filename, town, spectator_loc

# def main():
#    routes = read_routes(abspath("data/routes_training.xml"))
#    scenarios,obj,raw_list = read_scenarios(abspath("data/all_towns_traffic_scenarios.json"))
#    # for i,s in enumerate(scenarios):
#    #    if s["town"]=="Town01" and s["scenario_type"]=="Scenario4":
#    #       print(i,s["waypoint"][:2])

#    # print(distance_point(1,1,7,1,-2.2,1,2))
#    for scenario in tqdm(scenarios[62:64]):
#       candidate,counter  = None,0
#       while candidate==None:
#          candidate = find_route_snippet(rng, scenario, routes, 1)
#          print(counter)
#          counter += 1
         
#       write_route(rng, candidate, routes, weather="Random")
#       write_scenario_json(scenario,obj,raw_list)
#       # write_scenario_py(scenario)
#    print("dtop")

# if __name__=="__main__":
#     create_random_scenario()
