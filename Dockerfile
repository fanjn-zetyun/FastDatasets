FROM registry.hd-02.alayanew.com:8443/alayanew-4fd285c4-c4f3-4e92-80ee-26169717cba8/fastdatasets:1.13

# 复制FastDatasets源码
COPY . .

RUN chmod +x /app/entrypoint.sh
