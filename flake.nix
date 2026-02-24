{
  description = "Dev shell approximating conda --file linux-64 export (py38 stack)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-21.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        # Your export is "platform: linux-64" -> x86_64-linux
        _ = assert system == "x86_64-linux"; null;

        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true; # needed for mkl
          };
        };

        lib = pkgs.lib;

        python = pkgs.python38;

        py = python.withPackages (
          ps: let
            opt = name: lib.optionals (builtins.hasAttr name ps) [ps.${name}];
            optQ = name: lib.optionals (builtins.hasAttr name ps) [ps."${name}"];
          in
            (with ps; [
              # Python deps from your list
              certifi
              cycler
              kiwisolver
              matplotlib
              numpy
              olefile
              pandas
              pillow
              pyparsing
              pyqt5
              python-dateutil
              pytz
              scipy
              seaborn
              sip
              six
              tornado
              pip
              setuptools
              wheel
            ])
            # Optional / may not exist in this nixpkgs:
            ++ opt "pickle5"
            ++ optQ "mkl-service"
            ++ optQ "mkl_fft"
            ++ optQ "mkl_random"
        );

        sysLibs = with pkgs; [
          # ca-certificates
          cacert

          # dbus / core libs
          dbus
          expat
          fontconfig
          freetype
          glib
          icu
          pcre
          ncurses
          readline
          sqlite
          tk
          xz
          zlib
          zstd
          libffi
          libxml2
          libedit
          libuuid
          lz4

          # image stack (pillow/matplotlib)
          libjpeg
          lcms2
          libpng
          libtiff

          # gstreamer bits from your list
          gst_all_1.gstreamer
          gst_all_1.gst-plugins-base

          # qt + xcb bits (pyqt / qt platform plugin)
          qt5.qtbase
          xorg.libxcb
          xorg.libX11
          xorg.libXext
          xorg.libXrender
          xorg.libSM
          xorg.libICE
          xorg.libXi
          xorg.xcbutil
          xorg.xcbutilimage
          xorg.xcbutilkeysyms
          xorg.xcbutilrenderutil
          xorg.xcbutilwm

          # "ld_impl_linux-64" analogue
          binutils

          # your env used openssl 1.1.x; nixpkgs 21.05 has 1.1 available
          openssl_1_1
        ];

        # Optional MKL/OpenMP (maps your blas/mkl/intel-openmp lines).
        # Note: mkl is unfree and often not cached.
        mklStack = with pkgs; [
          mkl
          llvmPackages.openmp
        ];
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [py] ++ sysLibs ++ mklStack;

          nativeBuildInputs = [
            pkgs.makeWrapper
            pkgs.bashInteractive
            pkgs.qt5.wrapQtAppsHook
          ];

          shellHook = ''
            # Help Python TLS libs find certs
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

            # If you want matplotlib to use Qt explicitly:
            export MPLBACKEND=Qt5Agg

            # If you want MKL to use LLVM OpenMP, Intel suggests LD_PRELOAD.
            # export LD_PRELOAD=${pkgs.llvmPackages.openmp}/lib/libomp.so

            # Wrap an interactive bash with qtWrapperArgs so PyQt finds platform plugins (xcb).
            bashdir=$(mktemp -d)
            makeWrapper "$(type -p bash)" "$bashdir/bash" "''${qtWrapperArgs[@]}"
            exec "$bashdir/bash"
          '';
        };
      }
    );
}
