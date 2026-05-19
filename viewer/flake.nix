{
  description = "Dev shell for EEG .npy viewer (PySide6 / PyQtGraph / numpy / pandas)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11"; # pick the channel you prefer
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true; # optional if any dependencies are unfree
        };
      };

      # Python dev environment with required packages
      pythonEnv = pkgs.python313.withPackages (ps:
        with ps; [
          numpy
          pandas
          pyqtgraph
          pyside6
          # validation pipeline
          matplotlib
          seaborn
          scipy
          mne
        ]);
    in {
      devShells.default = pkgs.mkShell {
        name = "eeg-viewer-devshell";

        buildInputs = [
          pythonEnv
        ];

        # Helpful tools in your shell
        nativeBuildInputs = [
          pkgs.git
          pkgs.gnumake
        ];

        # Optional: environment vars
        shellHook = ''
          echo "EEG viewer devshell active!"
          echo "Python interpreter: $(which python)"
        '';
      };
    });
}
