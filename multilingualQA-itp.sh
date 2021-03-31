COMMIT=$1
script=$2
DATA_DIR=$3
RESULT_DIR=$4
NUM_EPOCHS=$5
LR=$6
SEED=$7




cd /tmp/code
if [ -d "apex" ]; then
  echo "apex installed in this machine"
else
  echo "apex not installed in this machine"
  echo 'installing apex'
  git clone https://github.com/NVIDIA/apex
  cd apex
  ls  
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
fi


cd /tmp/code
git clone https://github.com/huggingface/transformers
cd transformers
git checkout b90745c5901809faef3136ed09a689e7d733526c
pip install -e .
cd ..


cd /tmp/code
if [ -d "code" ]; then
  echo "code installed in this machine"
  cd code
else
  echo "code not installed in this machine"
  echo 'Prepare Code'
  ls
  git clone git@github.com:KyGao/Multilingual-SQuAD.git code
  cd code
  git checkout --force $COMMIT
fi




# add some system commands here as you wish
pip install numpy
pip install tensorboard
pip install tensorboardX
pip install matplotlib
export MKL_SERVICE_FORCE_INTEL=1
echo 'Solving MKL done!'
pip install --user --editable .

echo 'Showing data directory'
echo $DATA_DIR

export MKL_SERVICE_FORCE_INTEL=1
bash ${script} $DATA_DIR $RESULT_DIR $NUM_EPOCHS $LR $SEED