#!groovy

pipeline {
    agent any

    options {
        buildDiscarder(
            logRotator(
                daysToKeepStr: "21",
                numToKeepStr: "50",
                artifactDaysToKeepStr: "21",
                artifactNumToKeepStr: "50"
            )
        )
    }
    stages {
        stage('checkout') {
            steps {
                dir('octotiger') {
                    checkout scm
                    echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                }
            }
        }
        stage('build') {
            matrix {
                axes {
                    axis {
                        name 'configuration_name'
                        values 'gcc-8', 'gcc-9', 'clang-9', 'gcc-9-cuda-11', 'hipcc'
                    }
                    axis {
                         name 'build_type'
                         values 'Release', 'Debug'
                    }
                }
                stages {
                    stage('build') {
                        steps {
                            dir('hpx') {
                                sh '''
                                #!/bin/bash -l
                                .jenkins/lsu/entry.sh
                                '''
                            }
                        }
                    }
                }
            }
        }
    }
}
