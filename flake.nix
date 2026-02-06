{
  description = "Species perturbation experiments for TEA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python313;

        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          pandas
          scipy
          scikit-learn
          matplotlib
          pyyaml
          requests
          transformers
          accelerate
          datasets
          evaluate
          torchWithCuda
          nltk
          wordfreq
          pytest
          icecream
        ]);

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn

            # Common tools
            pkgs.git
          ];

          shellHook = ''
            set -e

            # HuggingFace cache
            export HF_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"
            export TOKENIZERS_PARALLELISM="false"

            # NLTK data storage
            export NLTK_DATA="''${XDG_CACHE_HOME:-$HOME/.cache}/nltk_data"

            echo "READY."
          '';
        };
      }
    );
}
