

import unittest
import pandas as pd
import numpy as np
from price_prediction import fetch_data, preprocess_data, build_and_train_model, make_predictions

class TestPricePrediction(unittest.TestCase):

    def setUp(self):
        # Mock data to be used in tests
        data = {
            'Close': np.random.rand(100)  # 100 random close prices
        }
        self.df = pd.DataFrame(data)

    def test_fetch_data(self):
        df = fetch_data('ETH-USD', '5m', '1d')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_preprocess_data(self):
        x, y, scaler = preprocess_data(self.df)
        self.assertEqual(x.shape[1], 60)
        self.assertEqual(x.shape[2], 1)
        self.assertEqual(len(x), len(y))

    def test_build_and_train_model(self):
        x, y, scaler = preprocess_data(self.df)
        split = int(0.8 * len(x))
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        model, history = build_and_train_model(x_train, y_train, x_val, y_val)
        self.assertEqual(model.output_shape, (None, 1))
        self.assertIn('loss', history.history)

    def test_make_predictions(self):
        x, y, scaler = preprocess_data(self.df)
        split = int(0.8 * len(x))
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        x_test, y_test = x_val, y_val  # Using validation data for test
        model, _ = build_and_train_model(x_train, y_train, x_val, y_val)
        predictions = make_predictions(model, x_test, scaler)
        self.assertEqual(predictions.shape, (len(x_test), 1))

if __name__ == "__main__":
    unittest.main()
