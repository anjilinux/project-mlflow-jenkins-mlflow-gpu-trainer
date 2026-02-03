pipeline {
    agent { label 'gpu-node' }

    environment {
        IMAGE_NAME = "mlflow-gpu-trainer"
        MLFLOW_TRACKING_URI = "http://host.docker.internal:5555"
    }

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Verify GPU on Jenkins Node') {
            steps {
                sh '''
                echo "===== Host GPU Info ====="
                nvidia-smi
                '''
            }
        }

        stage('Build GPU Image') {
            steps {
                sh '''
                docker build \
                --pull \
                --tag $IMAGE_NAME \
                .
                '''
            }
        }

        stage('GPU Sanity Check (Docker)') {
            steps {
                sh '''
                echo "===== GPU inside Docker ====="
                docker run --rm --gpus all \
                nvidia/cuda:13.1.1-base-ubuntu24.04 nvidia-smi
                '''
            }
        }

        stage('Check MLflow Server') {
            steps {
                sh '''
                echo "Checking MLflow availability..."
                curl -f $MLFLOW_TRACKING_URI || exit 1
                '''
            }
        }

        stage('Train Model (GPU + MLflow)') {
            steps {
                sh '''
                docker run --rm --gpus all \
                --add-host=host.docker.internal:host-gateway \
                -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
                -v $WORKSPACE:/workspace \
                $IMAGE_NAME \
                python -u /workspace/train.py
                '''
            }
        }
    }

    post {
        success {
            echo "🚀 Training completed and logged to MLflow"
        }
        failure {
            echo "❌ Pipeline failed"
        }
        always {
            sh 'docker system prune -f || true'
        }
    }
}
