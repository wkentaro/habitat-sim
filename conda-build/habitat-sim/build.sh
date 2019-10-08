#!/bin/bash


build_args=()
if [ ${HEADLESS} == "1" ]; then
  build_args+=("--headless")
fi

if [ ${WITH_CUDA} = "1" ]; then
  build_args+=("--with-cuda")
fi

if [ ${WITH_BULLET} == "1" ]; then
  build_args+=("--with-bullet")
fi

if [ $(uname) == "Linux" ]; then
  export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/usr/lib/x86_64-linux-gnu
  cp -r /usr/include/EGL ${PREFIX}/include/.
  cp -r /usr/include/X11 ${PREFIX}/include/.
fi

${PYTHON} setup.py install "${build_args[@]}"


if [ -f "build/viewer" ]; then
  cp -v build/viewer ${PREFIX}/bin/habitat-viewer
fi

pushd ${PREFIX}

corrade_bindings=$(find . -name "*_corrade*so")
magnum_bindings=$(find . -name "*_magnum*so")
hsim_bindings=$(find . -name "*habitat_sim_bindings*so")
ext_folder=$(dirname ${corrade_bindings})/habitat_sim/_ext

if [ $(uname) == "Darwin" ]; then
  install_name_tool -add_rpath @loader_path/habitat_sim/_ext ${corrade_bindings}
  install_name_tool -add_rpath @loader_path/habitat_sim/_ext ${magnum_bindings}
  install_name_tool -add_rpath @loader_path ${hsim_bindings}

  
  install_name_tool -add_rpath @loader_path/../${ext_folder} bin/habitat-viewer
  

  find $(dirname ${hsim_bindings}) -name "*Corrade*dylib" | xargs -I {} install_name_tool -add_rpath @loader_path {}


  pushd $(find . -name "corrade" -type d)
  find . -name "*so" | xargs -I {} install_name_tool -add_rpath @loader_path/../habitat_sim/_ext {}
  popd
elif [ $(uname) == "Linux" ]; then
  patchelf --set-rpath "\$ORIGIN/habitat_sim/_ext" ${corrade_bindings}
  patchelf --set-rpath "\$ORIGIN/habitat_sim/_ext" ${magnum_bindings}
  patchelf --set-rpath "\$ORIGIN" ${hsim_bindings}
  
  if [ -f 'bin/habitat-viewer' ]; then
    patchelf --set-rpath \$ORIGIN/../${ext_folder} bin/habitat-viewer
  fi

  find $(dirname ${hsim_bindings}) -name "*Corrade*so" | xargs -I {} patchelf --set-rpath \$ORIGIN {}

  pushd $(find . -name "corrade" -type d)
  find . -name "*so" | xargs -I {} patchelf --set-rpath \$ORIGIN/../habitat_sim/_ext {}
  popd
fi


popd