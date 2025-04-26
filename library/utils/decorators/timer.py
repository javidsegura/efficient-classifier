
import time

def timer(process_name: str):
      def wrapper(func):
            def measurer(*args, **kwargs):
                  start_time = time.time()
                  result = func(*args, **kwargs)
                  end_time = time.time()
                  print(f"{process_name} took {end_time - start_time} seconds")
                  return result
            return measurer
      return wrapper