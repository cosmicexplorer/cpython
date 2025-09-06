#!/usr/bin/zsh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

. ./colors.zsh

function as-log {
  "$@" >&2
  echo >&2
}

function set-exe-rpath {
  local -r lib_dir="$1"
  local -r python_exe="$2"
  as-log cyan "setting rpath for ${python_exe}"
  local canonical_lib_dir
  realpath -e "$lib_dir" | IFS='' read -r canonical_lib_dir
  patchelf --set-rpath "$canonical_lib_dir" \
           --force-rpath \
           "$python_exe"
}

function create-exe-symlinks {
  local -r bin_dir="$1"
  as-log cyan 'creating symlinks (pip, python)'
  pushd "$bin_dir"
  ln -sfv pip3 pip
  ln -sfv python3 python
  popd
}

function quiet-unless-err {
  local -r output="$(mktemp --suffix=build-cmd)"
  if ! "$@" > "$output"; then
    local -r rc="$?"
    as-log red '! command failed:' "$@"
    <"$output" >&2
    return "$rc"
  fi
}

case "${DO_CLEAN:-}" in
  from-scratch)
    as-log light_red 'cleaning everything!!!'
    rm -rf build/ install/
    git clean -Xd --force --quiet
    ;;
  distclean)
    as-log light_red 'distclean!!!'
    make -C build distclean
    ;;
  clean)
    as-log light_red 'cleaning built outputs'
    make -C build clean-retain-profile
    ;;
  *)
    as-log light_gray 'nothing to clean'
    ;;
esac

case "${WITH_SHARED:-}" in
  n|no|N|NO)
    as-log light_purple 'disabling shared library output'
    declare -ra shared_args=( )
    declare -r set_rpath=n
    ;;
  *)
    as-log light_purple 'enabling shared library output'
    declare -ra shared_args=( --enable-shared )
    declare -r set_rpath=y
    ;;
esac

case "${WITH_STATIC:-}" in
  n|no|N|NO)
    as-log dark_gray 'not building static library'
    declare -ra static_args=( --without-static-libpython )
    ;;
  *)
    as-log light_gray 'building static library'
    declare -ra static_args=( --with-static-libpython )
    ;;
esac

case "${WITH_FREE_THREADING:-}" in
  n|no|N|NO)
    as-log light_purple 'no free threading (gil enabled)'
    declare -ra gil_args=( )
    ;;
  *)
    as-log light_purple 'enabling free threading (gil disabled)'
    declare -ra gil_args=( --disable-gil )
    ;;
esac

case "${WITH_OPTIMIZATIONS:-}" in
  y|yes|Y|YES)
    as-log cyan 'enabling build optimizations!!!'
    declare -ra opt_args=(
      --enable-optimizations=yes
      --with-lto=yes
      --with-strict-overflow
      --with-computed-gotos
      --with-tail-call-interp
    )
    ;;
  *)
    as-log light_gray 'disabling optimizations'
    declare -ra opt_args=(
      --enable-optimizations=no
      --with-lto=no
    )
    ;;
esac

declare -ra desired_features=(
  --enable-pystats
  --enable-ipv6
  --enable-big-digits=30
  --with-builtin-hashlib-hashes='md5,sha1,sha2,sha3,blake2'
  --with-hash-algorithm=fnv
  --with-mimalloc
  --with-pymalloc
  --with-c-locale-coercion
  --with-readline=readline
)

case "${DO_CONFIGURE:-}" in
  y|yes|Y|YES)
    as-log cyan 'configuring!'
    pushd build
    ../configure \
      CC=/usr/bin/gcc CXX=/usr/bin/g++ \
      LDFLAGS='-fuse-ld=mold' \
      CFLAGS='-std=gnu2y' CXXFLAGS='-std=gnu++2c' \
      --with-pkg-config=yes \
      --disable-test-modules \
      --enable-safety \
      "${static_args[@]}" \
      "${shared_args[@]}" \
      "${gil_args[@]}" \
      "${opt_args[@]}" \
      "${desired_features[@]}" \
      --prefix="$(realpath -e ../install)"
    popd
    ;;
esac

export MAKEFLAGS="${MAKEFLAGS:---jobs=$(nproc)}"

make -C build/ python

quiet-unless-err make -C build/ install

create-exe-symlinks ./install/bin

if [[ "$set_rpath" == 'y' ]]; then
  set-exe-rpath ./install/lib ./install/bin/python
fi
