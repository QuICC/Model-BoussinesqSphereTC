include(ConfigureBenchmarks)
quicc_add_benchmark(Explicit
  MODEL "BoussinesqSphereTC"
  WORKDIR "${CMAKE_BINARY_DIR}/${QUICC_CURRENT_MODEL_DIR}/TestSuite/Benchmarks"
  ARCHIVEDIR "${CMAKE_BINARY_DIR}/Models/_refdata"
  )

quicc_add_benchmark(Explicit
  MODEL "BoussinesqSphereTC"
  WORKDIR "${CMAKE_BINARY_DIR}/${QUICC_CURRENT_MODEL_DIR}/TestSuite/Benchmarks"
  ARCHIVEDIR "${CMAKE_BINARY_DIR}/Models/_refdata"
  VARIANTS "framework/parallel/algorithm:serial" "framework/timestepping/scheme:ImEx_PC2" "setup/model/split_equation:On"
  FILTER "algorithm" "split_equation"
  DATAFILTER "split_equation"
  )
