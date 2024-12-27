  #include <SDL.h>
  #include <stdio.h>

  int main(int argc, char* argv[]) {
      // Initialize SDL with the video subsystem
      if (SDL_Init(SDL_INIT_VIDEO) != 0) {
          fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
          return 1;
      }

      // Get the number of available video drivers
      int numDrivers = SDL_GetNumVideoDrivers();
      printf("Number of video drivers available: %d\n", numDrivers);

      // List all available video drivers
      for (int i = 0; i < numDrivers; ++i) {
          printf("Video driver #%d: %s\n", i, SDL_GetVideoDriver(i));
      }

      // Clean up and quit SDL
      SDL_Quit();

      return 0;
  }
