import joblib
import tensorflow as tf
import os

def load_all_models(output_dir='outputs'):
    models = {}
    # Load traditional ML models
    models['isolation_forest'] = joblib.load(f'{output_dir}/isolation_forest_model.joblib')
    models['svm'] = joblib.load(f'{output_dir}/oneclass_svm_model.joblib')
    models['elliptic'] = joblib.load(f'{output_dir}/elliptic_envelope_model.joblib')
    models['lof'] = joblib.load(f'{output_dir}/lof_model.joblib')
    
    # Load AutoEncoder and its threshold
    models['autoencoder'] = tf.saved_model.load(os.path.join(output_dir, 'autoencoder_savedmodel'))
    models['ae_threshold'] = joblib.load(os.path.join(output_dir, 'autoencoder_threshold.joblib'))
    
    print("✅ All models loaded successfully.")
    return models

if __name__ == "__main__":
    models = load_all_models()
