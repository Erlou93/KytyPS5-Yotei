set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
set(KYTY_SHADER_REVISION "unknown")
if(GIT_EXECUTABLE)
	# The shader caches' identity (11.11): the last commit that changed the shader translator and the code that turns
	# its output into programs and pipelines, so a build that changes only other code keeps the translations.
	set(KYTY_SHADER_PATHS ":/src/graphics/shader" ":/src/graphics/host_gpu/renderer/pipeline" ":/src/graphics/host_gpu/shaders")
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" log -1 --format=%H -- ${KYTY_SHADER_PATHS}
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_SHADER_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE SHADER_REV_RESULT
		ERROR_QUIET
	)
	if(NOT SHADER_REV_RESULT EQUAL 0 OR KYTY_SHADER_REVISION STREQUAL "")
		set(KYTY_SHADER_REVISION "unknown")
	else()
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD -- ${KYTY_SHADER_PATHS}
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE SHADER_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT SHADER_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_SHADER_REVISION "-dirty")
		endif()
	endif()
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT GIT_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
	endif()
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
