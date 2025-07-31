# Get Joern and the python dataset
# the csv is stored in a google file store at 
# https://drive.google.com/file/d/14vtngKXaBPI43aKRfd6-PoV3peDtd5XU/view?usp=sharing
# cd ..

if [[ -d sourcescripts/storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents sourcescripts/storage/external
fi

cd sourcescripts/storage/external

if [[ ! -d joern-cli ]]; then
    wget https://github.com/joernio/joern/releases/download/v2.0.331/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi

# if [[ ! -f "domain_Dataset-Python.csv" ]]; then
#     gdown https://drive.google.com/uc\?id\=14vtngKXaBPI43aKRfd6-PoV3peDtd5XU
#     unzip domain_Dataset-Python.zip
#     rm domain_Dataset-Python.zip
# else
#     echo "Already downloaded Python version of the Dataset data"
# fi


# https://drive.google.com/file/d/14TT--A5nFHmbNSe3vfqmz930RDAhKFO1/view?usp=sharing
# https://drive.google.com/file/d/1YdWbFtReHK0nBkfjY9OrLFG68sEkyxYM/view?usp=sharing

if [[ ! -f "MegaVul_Java_Domain.csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1YdWbFtReHK0nBkfjY9OrLFG68sEkyxYM
    unzip MegaVul_Java_Domain.zip
    rm MegaVul_Java_Domain.zip
else
    echo "Already downloaded Megavul version of the Dataset data"
fi


cd ..
mkdir cache

cd cache

if [[ ! -f "codebert_finetuned" ]]; then
    gdown https://drive.google.com/uc?id=1X-QitxtD3Djdg8lJNrzkyzomBSwHu-A1
    unzip codebert_finetuned.zip
    rm codebert_finetuned.zip
else
    echo "Already downloaded codebert finetuned"
fi
