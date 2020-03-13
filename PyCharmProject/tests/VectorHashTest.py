import unittest
import app.VectorUtils as VectorUtils


class CorrectVectorHashingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.vector1 = [1.234567, -0.123456, 0.1234567, -1.234567]
        self.vector2 = [-1.234567, -0.123456, 0.1234567, -1.234567]
        self.vector3 = [-1.234566, -0.123457, 0.1234566, -1.234566]
        self.vector4 = [-1.234547, -0.123446, 0.1234547, -1.234547]

    def test_hash1(self):
        vector1_hash = VectorUtils.hash_vector1(self.vector1)
        vector2_hash = VectorUtils.hash_vector1(self.vector2)
        vector3_hash = VectorUtils.hash_vector1(self.vector3)
        vector4_hash = VectorUtils.hash_vector1(self.vector4)

        self.assertNotEqual(vector1_hash, vector2_hash)
        self.assertEqual(vector2_hash, vector3_hash)
        self.assertNotEqual(vector2_hash, vector4_hash)

    def test_hash2(self):
        vector1_hash = VectorUtils.hash_vector2(self.vector1)
        vector2_hash = VectorUtils.hash_vector2(self.vector2)
        vector3_hash = VectorUtils.hash_vector2(self.vector3)
        vector4_hash = VectorUtils.hash_vector2(self.vector4)

        self.assertNotEqual(vector1_hash, vector2_hash)
        self.assertEqual(vector2_hash, vector3_hash)
        self.assertNotEqual(vector2_hash, vector4_hash)

    def test_hash3(self):
        vector1_hash = VectorUtils.hash_vector3(self.vector1)
        vector2_hash = VectorUtils.hash_vector3(self.vector2)
        vector3_hash = VectorUtils.hash_vector3(self.vector3)
        vector4_hash = VectorUtils.hash_vector3(self.vector4)

        self.assertNotEqual(vector1_hash, vector2_hash)
        self.assertEqual(vector2_hash, vector3_hash)
        self.assertNotEqual(vector2_hash, vector4_hash)

    def test_hash4(self):
        vector1_hash = VectorUtils.hash_vector4(self.vector1)
        vector2_hash = VectorUtils.hash_vector4(self.vector2)
        vector3_hash = VectorUtils.hash_vector4(self.vector3)
        vector4_hash = VectorUtils.hash_vector4(self.vector4)

        self.assertNotEqual(vector1_hash, vector2_hash)
        self.assertEqual(vector2_hash, vector3_hash)
        self.assertNotEqual(vector2_hash, vector4_hash)

    def test_hash5(self):
        vector1_hash = VectorUtils.hash_vector5(self.vector1)
        vector2_hash = VectorUtils.hash_vector5(self.vector2)
        vector3_hash = VectorUtils.hash_vector5(self.vector3)
        vector4_hash = VectorUtils.hash_vector5(self.vector4)

        self.assertNotEqual(vector1_hash, vector2_hash)
        self.assertEqual(vector2_hash, vector3_hash)
        self.assertNotEqual(vector2_hash, vector4_hash)

    def test_hash6(self):
        vector1_hash = VectorUtils.hash_vector6(self.vector1)
        vector2_hash = VectorUtils.hash_vector6(self.vector2)
        vector3_hash = VectorUtils.hash_vector6(self.vector3)
        vector4_hash = VectorUtils.hash_vector6(self.vector4)

        self.assertNotEqual(vector1_hash, vector2_hash)
        self.assertEqual(vector2_hash, vector3_hash)
        self.assertNotEqual(vector2_hash, vector4_hash)



if __name__ == '__main__':
    unittest.main()