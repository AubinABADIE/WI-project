import Dependencies._

ThisBuild / scalaVersion := "2.12.10"

lazy val root = (project in file("."))
  .settings(
    name:="WI-Project",
    libraryDependencies ++= Seq(
      spark,
      sparkSql
    )
  )