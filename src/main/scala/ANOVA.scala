import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ANOVA {
/**
  * Create a class, CatTuple, to pass to the ANOVA function so that columns can be referred to by specific names.
  * Create a class, ANOVAStats, that will be returned from the ANOVA function so that its outputs can be selected and referred to by name.
  **/
final case class CatTuple(cat: String, value: Double)
final case class ANOVAStats(dfb: Long, dfw: Double, F_value: Double, etaSq: Double, omegaSq: Double)

// Column names to use when converting to CatTuple
val colnames = Seq("cat", "value")

/**
  * Implementation of ANOVA function: calculates the degrees of freedom, F-value, eta squared and omega squared values.
  * Utilizes Spark 2.0's RelationalGroupedDataset instead of Iterable[RDD[Double]]
  **/
def getAnovaStats(categoryData: org.apache.spark.sql.Dataset[CatTuple], spark : SparkSession) : ANOVAStats = {
  import spark.implicits._

  categoryData.createOrReplaceTempView("df")
  val newdf = spark.sql("select A.cat, A.value, cast((A.value * A.value) as double) as valueSq, ((A.value - B.avg) * (A.value - B.avg)) as diffSq from df A join (select cat, avg(value) as avg from df group by cat) B where A.cat = B.cat")
  val grouped = newdf.groupBy("cat")
  val sums = grouped.sum("value")
  val counts = grouped.count
  val numCats = counts.count
  val sumsq = grouped.sum("valueSq")
  val avgs = grouped.avg("value")

  val totN = counts.agg(org.apache.spark.sql.functions.sum("count")).
    first.get(0) match {case d: Double => d case l: Long => l.toDouble}
  val totSum = sums.agg(org.apache.spark.sql.functions.sum("sum(value)")).
    first.get(0) match {case d: Double => d case l: Long => l.toDouble}
  val totSumSq = sumsq.agg(org.apache.spark.sql.functions.sum("sum(valueSq)")).
    first.get(0) match {case d: Double => d case l: Long => l.toDouble}

  val totMean = totSum / totN

  val dft = totN - 1
  val dfb = numCats - 1
  val dfw = totN - numCats

  val joined = counts.as("a").join(sums.as("b"), $"a.cat" === $"b.cat").join(sumsq.as("c"),$"a.cat" === $"c.cat").join(avgs.as("d"),$"a.cat" === $"d.cat").select($"a.cat",$"count",$"sum(value)",$"sum(valueSq)",$"avg(value)")
  val finaldf = joined.withColumn("totMean", lit(totMean))

  val ssb_tmp = finaldf.rdd.map(x => (x(0).asInstanceOf[String], (((x(4)  match {case d: Double => d case l: Long => l.toDouble}) - (x(5) match {case d: Double => d case l: Long => l.toDouble})) * ((x(4) match {case d: Double => d case l: Long => l.toDouble}) - (x(5) match {case d: Double => d case l: Long => l.toDouble})))* (x(1) match {case d: Double => d case l: Long => l.toDouble})))
  val ssb = ssb_tmp.toDF.agg(org.apache.spark.sql.functions.sum("_2")).first.get(0) match {case d: Double => d case l: Long => l.toDouble}

  val ssw_tmp = grouped.sum("diffSq")
  val ssw = ssw_tmp.agg(org.apache.spark.sql.functions.sum("sum(diffSq)")).first.get(0) match {case d: Double => d
  case l: Long => l.toDouble}

  val sst = ssb + ssw

  val msb = ssb / dfb
  val msw = ssw / dfw
  val F = msb / msw

  val etaSq = ssb / sst
  val omegaSq = (ssb - ((numCats - 1) * msw))/(sst + msw)

  ANOVAStats(dfb, dfw, F, etaSq, omegaSq)
}


//  override def main(args: Array[String]) {
//  /**
//    * Example of how to convert to CatTuple, call the ANOVA function and return the F-value
//    **/
//  //Ignores logs that aren't errors
//  Logger.getLogger("org").setLevel(Level.ERROR)
//  Logger.getLogger("akka").setLevel(Level.ERROR)
//
//  val conf = new SparkConf().setAppName("Cortex").setMaster("local")
//  val sc = new SparkContext(conf)
//
//  //Creates spark session
//  val spark = SparkSession
//    .builder
//    .master("local")
//    .appName("Cortex")
//    .getOrCreate
//
//  import spark.implicits._ // << add this after defining spark
//
//  val df = spark
//    .read
//    .option("header", "true")
//    .option("delimiter", ",")
//    .option("inferSchema", "true")
//    .csv("data/exportCSVANOVA")
//    .drop("interests")
//
//
//    val df2 = df.
//      withColumn("label", udf((elem: Double) =>
//        if(elem == 1.0)
//          "true"
//        else
//          "false"
//      ).apply(col("label")))
//
//  //val featuresCols = Array("bidfloor","exchange","appOrSite", "city", "media", "network", "os", "publisher", "type", "user")
//  val catTuple = df2.select("label", "user")
//    .withColumnRenamed("label", "cat")
//    .withColumnRenamed("user", "value").as[CatTuple]
//
//  val anovaStats = getAnovaStats(catTuple, spark)
//
//  print(anovaStats.F_value)
//  spark.close()
//}
}