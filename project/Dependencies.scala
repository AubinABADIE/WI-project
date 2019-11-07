import sbt._

object Dependencies {
  lazy val spark = "org.apache.spark" %% "spark-core" % "2.4.4"
  lazy val sparkSql = "org.apache.spark" %% "spark-sql" % "2.4.4"
  lazy val sparkMLlib = "org.apache.spark" %% "spark-mllib" % "2.4.0"
}