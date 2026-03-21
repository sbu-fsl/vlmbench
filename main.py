import sys

from src.vlmbench import VLMBench

if __name__ == "__main__":
    # create a VLMBench instance
    instance = VLMBench()

    try:
        # run the root command
        instance.run()
    except KeyboardInterrupt:
        print("VLMBench stopped by user.")

        # on user interrupt, safe shutdown and exit
        instance.shutdown()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

        # on any errors, safe shutdown and exit with non-zero
        instance.shutdown()
        sys.exit(1)
