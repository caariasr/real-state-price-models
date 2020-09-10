# real-state-price-models API

## Instruction to build and run the docker image

Navigate to the `api` folder, and then execute

```
docker build -t avm-predict .
docker run -p 127.0.0.1:8080:8080  avm-predict
```

After that you can test the API using your favorite tool. I add here a web
[link](http://127.0.0.1:8080/predict?offering_type_id=1&bedroom_id=3&bathroom_id=4&property_sqft=1794&property_cheques=1&meta_valid_from_dts=2018-10-27%2017%3A36%3A30&meta_valid_to_dts=2018-11-20%2011%3A33%3A25&coordinates_lat=%2025.0603&coordinates_lon=55.1793)
request example that can be used directly in your favorite browser. As you
can see from the link, the parameters are each of the relevant features. The
additional features are engineered before returning the corresponding price
prediction.



