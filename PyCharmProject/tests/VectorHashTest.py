import unittest
import TimeHashing

class CorrectVectorHashingTest(unittest.TestCase):
    vector1 = [1.23456,-0.12345,0.123456,-1.23456]
    vector2 = [-1.23456,-0.12345,0.123456,-1.23456]
    vector3 = [-1.23457,-0.12344,0.123455,-1.23455]

    vector1_hash = TimeHashing.hash_vector1(vector1)
    vector2_hash = TimeHashing.hash_vector1(vector2)

    self.assertNotEqual(vector1_hash, vector2_hash)
    self.assertEqual(vector2_hash, vector3_hash)


if __name__ == '__main__':
    unittest.main()