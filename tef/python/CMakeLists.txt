find_program(PYTHON "python")

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(BUILD_PACKAGE_TASK      "${CMAKE_CURRENT_SOURCE_DIR}/timestamp")


configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT  ${BUILD_PACKAGE_TASK}
                   COMMAND ${PYTHON} ${SETUP_PY} bdist_wheel
                   COMMENT "building tef package")

add_custom_target(tef ALL DEPENDS tef_core ${BUILD_PACKAGE_TASK})


