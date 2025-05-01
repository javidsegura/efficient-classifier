
import time

def timer(process_name: str):
      def wrapper(func):
            def measurer(*args, **kwargs):
                  print(f"=> Running {process_name}...")
                  start_time = time.time()
                  result = func(*args, **kwargs)
                  end_time = time.time()
                  print(f"\t=> {process_name} took {end_time - start_time} seconds")
                  return result
            return measurer
      return wrapper