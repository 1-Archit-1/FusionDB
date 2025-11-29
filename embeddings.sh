BASE="./directory"
EMBEDDINGS="$BASE/embeddings"
METADATA="$BASE/metadata"
mkdir -p "$EMBEDDINGS" "$METADATA"

for number in {0..10}
do
    wget --tries=100 -P "$EMBEDDINGS" https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${number}.npy          # download image embedding
    wget --tries=100 -P "$EMBEDDINGS" https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_${number}.npy        # download text embedding
    wget --tries=100 -P "$METADATA" https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_${number}.parquet    # download metadata
done


