# # import unittest
# # import warnings
# # import numpy as np
# # from PIL import Image
# # from io import BytesIO
# # from unittest.mock import patch, MagicMock

# # # Mock the load_model and pickle.load functions before importing the app
# # with patch('app.app.load_model') as mock_load_model, patch('app.app.pickle.load') as mock_pickle_load:
# #     # Mock the load_model function to return a mock model
# #     mock_model = MagicMock()
# #     mock_load_model.return_value = mock_model

# #     # Mock the pickle.load function to return a mock recommendation model
# #     mock_recommendation_model = MagicMock()
# #     mock_pickle_load.return_value = mock_recommendation_model

# #     from app.app import app, preprocess_image

# # class TestApp(unittest.TestCase):

# #     def setUp(self):
# #         self.app = app.test_client()
# #         self.app.testing = True

# #     def test_preprocess_image(self):
# #         # Create a dummy image
# #         image = Image.new('RGB', (256, 256), color='red')
# #         processed_image = preprocess_image(image)
        
# #         # Check the shape of the processed image
# #         self.assertEqual(processed_image.shape, (1, 224, 224, 3))
# #         # Check the type of the processed image
# #         self.assertIsInstance(processed_image, np.ndarray)
# #         # Check the pixel value range
# #         self.assertTrue(np.all(processed_image <= 1.0) and np.all(processed_image >= 0.0))

# #     def test_predict_no_file(self):
# #         response = self.app.post('/predict', data={})
# #         self.assertEqual(response.status_code, 400)
# #         self.assertIn(b'No file part', response.data)

# #     def test_predict(self):
# #         # Suppress TensorFlow warnings
# #         warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
        
# #         # Create a dummy image
# #         image = Image.new('RGB', (256, 256), color='red')
# #         img_byte_arr = BytesIO()
# #         image.save(img_byte_arr, format='PNG')
# #         img_byte_arr = img_byte_arr.getvalue()

# #         response = self.app.post('/predict', data={'file': (BytesIO(img_byte_arr), 'test.png')})
# #         self.assertEqual(response.status_code, 200)
# #         self.assertIn(b'prediction', response.data)

# #     def test_recommend_crop_missing_fields(self):
# #         response = self.app.post('/recommend', json={})
# #         self.assertEqual(response.status_code, 400)
# #         self.assertIn(b'Missing required fields', response.data)

# #     def test_recommend_crop(self):
# #         # Suppress scikit-learn warnings
# #         warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        
# #         data = {
# #             'N': 90,
# #             'P': 42,
# #             'K': 43,
# #             'temperature': 20,
# #             'humidity': 80,
# #             'ph': 6.5,
# #             'rainfall': 200
# #         }
# #         response = self.app.post('/recommend', json=data)
# #         self.assertEqual(response.status_code, 200)
# #         self.assertIn(b'recommended_crop', response.data)
        
# #         # Additional assertions to verify the response content
# #         response_json = response.get_json()
# #         self.assertIn('recommended_crop', response_json)
# #         self.assertIsInstance(response_json['recommended_crop'], str)

# # if __name__ == '__main__':
# #     unittest.main()
    
    
    
    
    
# import unittest
# import pytest   
# import warnings
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from unittest.mock import patch, MagicMock

# # Mock the load_model and pickle.load functions before importing the app
# with patch('app.app.load_model') as mock_load_model, patch('app.app.pickle.load') as mock_pickle_load:
#     # Mock the load_model function to return mock models
#     mock_crop_disease_model = MagicMock()
#     mock_class_indices = {'0': 'Healthy', '1': 'Diseased'}
#     mock_crop_recommendation_model = MagicMock()
#     mock_load_model.return_value = (mock_crop_disease_model, mock_class_indices, mock_crop_recommendation_model)

#     # Mock the pickle.load function to return a mock recommendation model
#     mock_pickle_load.return_value = mock_crop_recommendation_model

#     from app.app import app, preprocess_image

# class TestApp(unittest.TestCase):

#     def setUp(self):
#         self.app = app.test_client()
#         self.app.testing = True

#     def test_preprocess_image(self):
#         # Create a dummy image
#         image = Image.new('RGB', (256, 256), color='red')
#         processed_image = preprocess_image(image)
        
#         # Check the shape of the processed image
#         self.assertEqual(processed_image.shape, (1, 224, 224, 3))
#         # Check the type of the processed image
#         self.assertIsInstance(processed_image, np.ndarray)
#         # Check the pixel value range
#         self.assertTrue(np.all(processed_image <= 1.0) and np.all(processed_image >= 0.0))

#     def test_predict_no_file(self):
#         response = self.app.post('/predict', data={})
#         self.assertEqual(response.status_code, 400)
#         self.assertIn(b'No file part', response.data)

#     def test_predict(self):
#         # Suppress TensorFlow warnings
#         warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
        
#         # Create a dummy image
#         image = Image.new('RGB', (256, 256), color='red')
#         img_byte_arr = BytesIO()
#         image.save(img_byte_arr, format='PNG')
#         img_byte_arr = img_byte_arr.getvalue()

#         response = self.app.post('/predict', data={'file': (BytesIO(img_byte_arr), 'test.png')})
#         self.assertEqual(response.status_code, 200)
#         self.assertIn(b'prediction', response.data)

#     def test_recommend_crop_missing_fields(self):
#         response = self.app.post('/recommend', json={})
#         self.assertEqual(response.status_code, 400)
#         self.assertIn(b'Missing required fields', response.data)

#     def test_recommend_crop(self):
#         # Suppress scikit-learn warnings
#         warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        
#         data = {
#             'N': 90,
#             'P': 42,
#             'K': 43,
#             'temperature': 20,
#             'humidity': 80,
#             'ph': 6.5,
#             'rainfall': 200
#         }
#         response = self.app.post('/recommend', json=data)
#         self.assertEqual(response.status_code, 200)
#         self.assertIn(b'recommended_crop', response.data)
        
#         # Additional assertions to verify the response content
#         response_json = response.get_json()
#         self.assertIn('recommended_crop', response_json)
#         self.assertIsInstance(response_json['recommended_crop'], str)

# if __name__ == '__main__':
#     unittest.main()


from app.app import app
import unittest
import warnings
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import patch, MagicMock

class TestApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_no_file(self):
        response = self.app.post('/predict', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No file part', response.data)

    def test_predict(self):
        # Suppress TensorFlow warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
        
        # Create a dummy image
        image = Image.new('RGB', (256, 256), color='red')
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        response = self.app.post('/predict', data={'file': (BytesIO(img_byte_arr), 'test.png')})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'prediction', response.data)

    def test_recommend_crop_missing_fields(self):
        response = self.app.post('/recommend', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Missing required fields', response.data)

    def test_recommend_crop(self):
        # Suppress scikit-learn warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        
        data = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20,
            'humidity': 80,
            'ph': 6.5,
            'rainfall': 200
        }
        response = self.app.post('/recommend', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'recommended_crop', response.data)
        
        # Additional assertions to verify the response content
        response_json = response.get_json()
        self.assertIn('recommended_crop', response_json)
        self.assertIsInstance(response_json['recommended_crop'], str)
        
if __name__ == '__main__':
    unittest.main()
    
    