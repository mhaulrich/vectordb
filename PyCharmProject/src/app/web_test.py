from app import app
import unittest
import json
import numpy as np
import time

class TestService(unittest.TestCase):
    
    test = app.test_client()
    databaseName = 'web_unit_test_table'
    databaseDims = 3
            
    def test_1_CreateDatabase(self):
        print("\nTesting Create table\n")
        #Make sure database dosn't already exist
        try:
            self.test.delete('/databases/%s'%self.databaseName)
        except:
            pass
        
        #Create database
        params = {'dbname': self.databaseName, 'dimensions': self.databaseDims}
        resp = self.test.post('/databases', data=params)
        self.assertEqual(201, resp.status_code, "Service should return 201 CREATED")
        
        #Create a second time and verify it breaks
        resp2 = self.test.post('/databases', data=params)
        self.assertEqual(409, resp2.status_code, "Service should return 409 CONFLICT when table exists")
        
        #Check response contains our table info
        data = json.loads(resp.data)
        self.assertEqual(self.databaseName, data['name'], "Service should respond the requested name")
        self.assertEqual(self.databaseDims, data['dimensions'], "Service should respond the requested dimensions")
        
    
    def test_2_AddPoint(self):
        #Add a single point
        point = (np.random.rand(1,self.databaseDims)*10).round().tolist()
        asset = ["singleAsset"]
        print("\nTesting adding one point (%s | '%s')\n"%(point,asset))
        params = {'vectors': point, 'assets': asset}
        resp = self.test.post('/databases/%s/'%self.databaseName, data=json.dumps(params), content_type='application/json')
        self.assertEqual(201, resp.status_code, "Service should return 201 CREATED")
        data = json.loads(resp.data)
        self.assertTrue(len(data)==1, "Service should return a list of 1 id, when one point is added")
        pointID = data[0]

        #Force VectorIndex to be flushed to disk so we can retrieve/search points
        # resp2 = self.test.post('/databases/%s/flush'%self.databaseName)
        # self.assertEqual(200, resp2.status_code, "Flush operation should retun 200 OK")
        #Or wait 2 secs for it to be done automatically:
        time.sleep(2)
        
        #Lookup the point and verify the asset is there
        resp3 = self.test.get('/databases/%s/points/%s/'%(self.databaseName, pointID))
        self.assertEqual(200, resp3.status_code, "Service should return OK on get existing point")
        data3 = json.loads(resp3.data)
        print(data3)
        self.assertEqual(pointID, data3['id'], "Service should respond json with the same ID as was looked up.")
        self.assertIn(asset[0], data3['assets'], "Asset '%s' should exist in JSON"%asset[0])
        self.assertListEqual(point[0], data3['vector'], "Vector '%s' should exist in JSON"%point[0])
        
        
    def test_3_AddMultiplePoints(self):
        n = 5
        points = (np.random.rand(n,self.databaseDims)*10).round().tolist()
        assets = ["assetNum%d"%i for i in range(n)]
        print("\nTesting adding %d points: \nPoints: %s\nAssets: %s \n"%(n,points,assets))
        
        params = {'vectors': points, 'assets': assets}
        resp = self.test.post('/databases/%s/'%self.databaseName, data=json.dumps(params), content_type='application/json')
        self.assertEqual(201, resp.status_code, "Service should return 201 CREATED")
        data = json.loads(resp.data)
        self.assertTrue(len(data)==n, "Service should return a list of %d ids, when %d points are added"%(n,n))
        
        for pointID,point,asset in zip(data,points,assets):
            pointResp = self.test.get('/databases/%s/points/%s/'%(self.databaseName, pointID))
            self.assertEqual(200, pointResp.status_code, "Service should return OK on get existing point")
            pointData = json.loads(pointResp.data)
            self.assertEqual(pointID, pointData['id'], "Service should respond json with the same ID as was looked up.")
            self.assertIn(asset,pointData['assets'],"Asset '%s' should exist in JSON"%asset)

        #Force VectorIndex to be flushed to disk so we can retrieve/search points
        # self.test.post('/databases/%s/flush'%self.databaseName)
        #Or wait 2 secs for it to be done automatically:
        time.sleep(2)
        
        #Test the pointlist
        pointListParams = {'count': n} #There should be at least n points
        pointListResp = self.test.get('/databases/%s/points/'%(self.databaseName), data=pointListParams)
        self.assertEqual(200, pointResp.status_code, "Service should return OK on get existing pointlist")
        pointListData = json.loads(pointListResp.data)
        self.assertEqual(n, len(pointListData), "Service should return a point list of the requested length")
        
        #Test that we can't add points with mismatching dimensions
        badpoints = (np.random.rand(n,self.databaseDims+1)*10).round().tolist()
        badparams = {'vectors': badpoints, 'assets': assets}
        badresp = self.test.post('/databases/%s/'%self.databaseName, data=json.dumps(badparams), content_type='application/json')
        self.assertEqual(400, badresp.status_code, "Service should return 400 BAD REQUEST")
        
        #Test that we can't add points with mismatching assets
        badassets = ["assetNum%d"%i for i in range(n-1)]
        badparams2 = {'vectors': points, 'assets': badassets}
        badresp2 = self.test.post('/databases/%s/'%self.databaseName, data=json.dumps(badparams2), content_type='application/json')
        self.assertEqual(400, badresp2.status_code, "Service should return 400 BAD REQUEST")
        
    def test_4_Lookup(self):
        n = 2
        querypoints = (np.random.rand(n,self.databaseDims)*10).round().tolist()
        print("\nTesting looking up %d points: \nPoints: %s\n"%(n,querypoints))
        
        params = {'vectors': querypoints, 'exact': False, 'count': 5}
        resp = self.test.post('/databases/%s/lookup/'%self.databaseName, data=json.dumps(params), content_type='application/json')
        respData = json.loads(resp.data)
        self.assertEqual(200, resp.status_code, "Service should return 200 OK upon lookup")
        self.assertTrue(len(respData)==n, "Service response should have same length as # querypoints")
        
        for respResult in respData:
            #Check that the queryID field is present:
            self.assertIn("queryid", respResult, "Service response result should have a queryid field")
            self.assertTrue(len(respResult['neighbours'])>0, "Neighbours should not be empty")
            for neighbour in respResult['neighbours']:
                self.assertIn("id", neighbour, "Neighbour should have an ID field")
                self.assertIn("distance", neighbour, "Neighbour should have a distance field")
                self.assertTrue(len(neighbour['assets'])>0, "Neighbour assets should not be empty")
    
    def test_5_DeleteDatabase(self):
        print("\nTesting Delete table\n")
        resp = self.test.delete('/databases/%s'%self.databaseName)
        self.assertEqual(204, resp.status_code, "Service should return 204 NO CONTENT upon delete")
        
        
    
if __name__ == '__main__':
    unittest.main()