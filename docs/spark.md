# Running Spark

The Spark stages (b and d) run on a Spark Connect server: Spark 4.2.0, Scala 2.13, Java 21, with GraphFrames 0.12.3 installed on the server. The same code also runs on a local Spark.

## Connect to the cluster

1. Use the project environment (`environment.yml`). The client needs `pyspark[connect]==4.2.0` and `graphframes-py==0.12.3`, and pandas below 3.
2. Put the server's address in `.env`, which Git ignores:
   ```
   SPARK_REMOTE=sc://<host>:<port>
   ```
   The address is not kept in this repo. The network path to the server must be up.
3. Check the connection from the repo root:
   ```
   python -m scripts.check_spark
   ```
   It prints the Spark version, PageRank on a five-node made-up graph, and the number of connected components, which must be 2.

With Spark Connect, jars such as GraphFrames live on the server. Adding a package on the client does nothing.

## Run Spark locally instead

Leave `SPARK_REMOTE` empty, install `pyspark==4.2.0`, use Java 17 or 21, and set:
```
SPARK_MASTER=local[*]
GRAPHFRAMES_PACKAGE=io.graphframes:graphframes-spark4_2.13:0.12.3
```

## Notes

- GraphFrames 0.12.3 is built against Spark 4.1.2. It passed the toy check on 4.2.0; watch for problems on larger graphs.
- `connectedComponents` needs a checkpoint directory. On the cluster it is set on the server. Locally, set one with `spark.sparkContext.setCheckpointDir(...)`.
