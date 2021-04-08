import configparser as cpr
from pyspark import SparkConf

def get_spark_app_config():
  spark_conf = SparkConf()
  config=cpr.ConfigParser()
  config.read("conf/spark.conf")
  
  for (key, val) in config.items("SPARK_APP_CONFIGS"):
    spark_conf.set(key, val)
  return spark_conf
