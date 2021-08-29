#!/bin/sh
mkdir test_assets
cd test_assets
wget "https://nextcloud.mpi-klsb.mpg.de/index.php/s/RfF3yWJBJxSEMLa/download?path=%2F&files=SUB4_MPI_Etage6_working_standing.pkl" -O SUB4_MPI_Etage6_working_standing.pkl
wget "https://nextcloud.mpi-klsb.mpg.de/index.php/s/9BPQTmXS3w8Fc4H/download?path=%2F&files=SUB4.json" -O SUB4.json
wget "https://nextcloud.mpi-klsb.mpg.de/index.php/s/Q6yNSLwaGbiEtD8/download?path=%2F&files=MPI_Etage6.zip" -O MPI_Etage6.zip
wget "https://nextcloud.mpi-klsb.mpg.de/index.php/s/jcZ8HFexbscb2qq/download" -O TRAJ_SUB4_MPI_Etage6_working_standing.json
