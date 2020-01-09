find_program(PYTHON "python")

set(SETUP_PY_IN "")
set(SETUP_PY    "")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT},
                   COMMAND ${PYTHON} ${SETUP_PY} build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT})

add_custom_target(tef ALL DEPENDS ${OUTPUT})


