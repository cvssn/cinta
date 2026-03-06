#!/usr/bin/env bash

set -e

echo "[info] iniciando o download das vozes experimentais..."

# path absoluto do diretório de script atual
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# diretório de destino relativo à localização deste script
TARGET_DIR="$SCRIPT_DIR/voices/streaming_model/experimental_voices"

echo "[info] diretório de script: $SCRIPT_DIR"
echo "[info] diretório alvo: $TARGET_DIR"

# certifica se o diretório alvo existe
echo "[info] criando diretório alvo caso necessário..."

mkdir -p "$TARGET_DIR"

# lista de arquivos e suas urls
FILES=(
    "experimental_voices_de.tar.gz|https://github.com/user-attachments/files/24035887/experimental_voices_de.tar.gz"
    "experimental_voices_fr.tar.gz|https://github.com/user-attachments/files/24035880/experimental_voices_fr.tar.gz"
    "experimental_voices_jp.tar.gz|https://github.com/user-attachments/files/24035882/experimental_voices_jp.tar.gz"
    "experimental_voices_kr.tar.gz|https://github.com/user-attachments/files/24035883/experimental_voices_kr.tar.gz"
    "experimental_voices_pl.tar.gz|https://github.com/user-attachments/files/24035885/experimental_voices_pl.tar.gz"
    "experimental_voices_pt.tar.gz|https://github.com/user-attachments/files/24035886/experimental_voices_pt.tar.gz"
    "experimental_voices_sp.tar.gz|https://github.com/user-attachments/files/24035884/experimental_voices_sp.tar.gz"
    "experimental_voices_en1.tar.gz|https://github.com/user-attachments/files/24189272/experimental_voices_en1.tar.gz"
    "experimental_voices_en2.tar.gz|https://github.com/user-attachments/files/24189273/experimental_voices_en2.tar.gz"
)

# baixa, extrai e limpa cada arquivo
for entry in "${FILES[@]}"; do
    IFS="|" read -r FNAME URL <<< "$entry"

    echo "[info] baixando $FNAME ..."
    wget -O "$FNAME" "$URL"

    echo "[info] extraindo $FNAME ..."
    tar -xzvf "$FNAME" -C "$TARGET_DIR"

    echo "[info] limpando $FNAME ..."
    rm -f "$FNAME"
done

echo "[sucesso] todas as caixas de som experimentais foram instaladas com sucesso!"
echo "[sucesso] os alto-falantes estão localizados em:"
echo "          $TARGET_DIR"