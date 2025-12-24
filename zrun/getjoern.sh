# Get Joern and the python dataset
# the csv is stored in a google file store at 
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

# if [[ ! -f "MegaVul_Java_Domain.csv" ]]; then
#     gdown https://drive.google.com/uc\?id\=1YdWbFtReHK0nBkfjY9OrLFG68sEkyxYM
#     unzip MegaVul_Java_Domain.zip
#     rm MegaVul_Java_Domain.zip
# else
#     echo "Already downloaded Megavul version of the Dataset data"
# fi


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
