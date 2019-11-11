import Dependencies._
import sbtassembly.AssemblyPlugin.defaultUniversalScript

ThisBuild / scalaVersion := "2.12.10"

lazy val root = (project in file("."))
  .settings(
    name:="WI-Project",
    libraryDependencies ++= Seq(
      spark,
      sparkSql,
      sparkMLlib
    )
  )

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) =>
    xs map {_.toLowerCase} match {
      case "manifest.mf" :: Nil | "index.list" :: Nil | "dependencies" :: Nil =>
        MergeStrategy.discard
      case ps @ x :: xs if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
        MergeStrategy.discard
      case "plexus" :: xs =>
        MergeStrategy.discard
      case "services" :: xs =>
        MergeStrategy.filterDistinctLines
      case "spring.schemas" :: Nil | "spring.handlers" :: Nil =>
        MergeStrategy.filterDistinctLines
      case _ => MergeStrategy.first
    }
  case "application.conf" => MergeStrategy.concat
  case "reference.conf" => MergeStrategy.concat
  case _ => MergeStrategy.first
}

//Generate executable
assemblyOption in assembly := (assemblyOption in assembly).value
  .copy(prependShellScript = Some(defaultUniversalScript(shebang = false)))

//Export to root dir
assemblyOutputPath in assembly := file(baseDirectory.value.getAbsolutePath+"/clickPredict.jar")