docker run --name fd-test \
  -e FASTDATASETS_PARAMS='{"api_key":"68b705e0-b16c-46c2-a4eb-44192c1b0fa8","base_url":"https://vllm-5V3w0wgP.test.llamafactory.online/v1","model_name":"ZGU78dZG4_CZsS1WRwI7l","input_files":["/app/tests/test.txt"],"output_dir":"/workspace/user-data/datasets","chunk_min_len":200,"chunk_max_len":1000,"questions_per_chunk":2,"output_formats":["alpaca","sharegpt"],"llm_concurrency":3,"file_concurrency":2}' \
   registry.hd-02.alayanew.com:8443/alayanew-4fd285c4-c4f3-4e92-80ee-26169717cba8/fastdatasets:1.9

FastDatasets/tests/AttentionIsAllYouNeed.pdf
  --network host \