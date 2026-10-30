import unittest
from travel_time import CAPACITY, speed_from_hourly_flow, travel_minutes
class TravelTimeTests(unittest.TestCase):
    def test_zero_flow_speed_limit_and_delay(self):
        self.assertEqual(speed_from_hourly_flow(0),60)
        self.assertAlmostEqual(travel_minutes(1,0,True),1.5)
        self.assertAlmostEqual(travel_minutes(1,0,False),1.0)
    def test_under_capacity_root(self):
        self.assertAlmostEqual(speed_from_hourly_flow(1000),50.475503,places=5)
        self.assertAlmostEqual(travel_minutes(1,1000,True),1.688695,places=5)
    def test_invalid_flow(self):
        for x in [-1, CAPACITY+1, float('nan')]:
            with self.assertRaises(ValueError):speed_from_hourly_flow(x)
    def test_invalid_distance_or_delay_metadata(self):
        with self.assertRaises(ValueError):travel_minutes(0,0,True)
        with self.assertRaises(ValueError):travel_minutes(1,0,None)

class CourseIntegrationTests(unittest.TestCase):
    def test_destination_flow_and_explicit_control(self):
        from graph import Edge
        from route_search import find_routes
        from travel_time import course_edge_minutes
        edges=[Edge('A','B',1,True)]
        result=find_routes(edges,'A','B',{'A':0,'B':1000},course_edge_minutes)
        self.assertAlmostEqual(result[0].minutes,1.688695,places=5)
    def test_missing_control_metadata_is_rejected(self):
        from graph import Edge
        from travel_time import course_edge_minutes
        with self.assertRaisesRegex(ValueError,'controlled'): course_edge_minutes(Edge('A','B',1),500)

if __name__=='__main__':unittest.main()
