import configparser as cpr
from pyspark import SparkConf

def get_spark_app_config():
  spark_conf = SparkConf()
  config=cpr.ConfigParser()
  config.read("conf/spark.conf")
  
  for (key, val) in config.items("SPARK_APP_CONFIGS"):
    spark_conf.set(key, val)
  return spark_conf

def load_survey_df(spark, url):
   spark.sparkContext.addFile(url)

   survey_df=spark.read \
   .format("csv") \
   .option("header", "true") \
   .option("inferSchema", "true") \
   .load('file://'+SparkFiles.get("sample.csv"))

   return survey_df

def count_by_country(survey_df):
  count_df= survey_df.select("Age", "Gender", "Country", "State") \
                     .where("Age <= 40") \
                     .groupBy("Country") \
                     .count()
  return count_df
