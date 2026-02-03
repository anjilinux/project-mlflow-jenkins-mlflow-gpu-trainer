pipeline {
    agent { label 'gpu-node' }

    environment {
        IMAGE_NAME = "mlflow-gpu-trainer"
        MLFLOW_TRACKING_URI = "http://localhost:5555"
    }

    stages {

        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Build GPU Image') {
            steps {
                sh 'nvidia-smi'
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('GPU Sanity Check') {
            steps {
                sh '''
                docker run --rm --gpus all \
                nvidia/cuda:13.1.1-base-ubuntu24.04 nvidia-smi
                '''
            }
        }

        stage('Check MLflow Server') {
            steps {
                sh 'curl -f $MLFLOW_TRACKING_URI'
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                docker run --rm --gpus all \
                --network host \
                -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
                -v $WORKSPACE:/workspace \
                $IMAGE_NAME \
                python -u /workspace/train.py
                '''
            }
        }
    }
}
