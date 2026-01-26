import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;

public class Wordle {
    public static Scanner s = new Scanner(System.in);
    public static final String[] WORD_LIST = {"hello", "crane"};
    public static final int TOTAL_TRIES = 4;
    public static void main(String[] args) {
        while (true) { 
            play();
            System.out.println("\nPlay Again?  y or n");
            if(s.next().toLowerCase().equals("n")){
                break;
            }
        }
        s.close();
        System.out.println("\nThanks for playing!");
    }

    public static void play() {
        String game_word = getRandomWord();

        HashSet<Character> availableChars = initAlphabet();
        HashSet<Character> wrongChars = new HashSet<>();

        boolean correctguess = false;
        int tries_counter = 0;
        String guess = "";

        System.out.println("Welcome to WORDLE!");
        while (!correctguess && tries_counter < TOTAL_TRIES) { 
            updateCharGuessData(availableChars, wrongChars, game_word, guess);
            displayGuessInfo(game_word, guess, tries_counter);

            guess = s.next();
            if(guess.toLowerCase().equals(game_word)){
                correctguess = true;
                break;
            }
            tries_counter++;
        }
        displayStats(game_word, correctguess, TOTAL_TRIES);
    }

    public static String getRandomWord(){
        long seed = System.currentTimeMillis() / 1000;  
        Random r = new Random(seed);

        int index = r.nextInt(WORD_LIST.length);
        assert index < WORD_LIST.length : "off by 1?";
        return WORD_LIST[index];
    }

    public static void displayStats(String game_word, boolean correctguess, int tries){
        System.out.printf(
            "%n%s | Total tries: %d %nGame Word: %s",
            correctguess ? "YOU WON!!!" : "You kinda suck tbh...",
            tries,
            game_word
        );
    }

    public static HashMap<Character, Integer> initCharMapCount(String game_word){
        HashMap<Character, Integer> hm = new HashMap<>();
        for(char c : game_word.toCharArray()){
            hm.put(c, hm.getOrDefault(c, 0) + 1);
        }

        return hm;
    }

    public static HashSet<Character> initAlphabet(){
        HashSet<Character> output = new HashSet<>();
        for(char c = 'a'; c <= 'z'; c++){
            output.add(c);
        }

        return output;
    }

    public static void updateCharGuessData(HashSet<Character> availableChars, HashSet<Character> wrongChars, String game_word, String guess){
        for(int i = 0; i < guess.length(); i++){
            if(game_word.indexOf(guess.charAt(i)) == -1){
                availableChars.remove(guess.charAt(i));
                wrongChars.add(guess.charAt(i));
            }
        }

        System.out.println("\nAvaiable characters: " + availableChars);
        System.out.println("Wrong characters:    " + wrongChars);
    }

    public static void displayGuessInfo(String game_word, String guess, int tries_counter){
        // GYXYG
        HashMap<Character, Integer> CharMapCount = initCharMapCount(game_word);
        String output = "";
        for(int i = 0; i < game_word.length(); i++){
            if(i > guess.length() - 1){ // out of bounds check
                output += "X";
                continue;
            }
            
            Character guessChar = guess.charAt(i);
            if(game_word.charAt(i) == guessChar) {
                output += "G";
                CharMapCount.put(guessChar, CharMapCount.get(game_word.charAt(i)) - 1);
                continue;
            }

            if(game_word.indexOf(guessChar) != -1){
                int lettercount = CharMapCount.get(guessChar);

                if(lettercount > 0){
                    output += "Y";
                    CharMapCount.put(guessChar, lettercount - 1);
                }
                continue;
            }

            output += "X";
        }
        System.out.println(output);
        System.out.println(guess + " You have " + (TOTAL_TRIES - tries_counter) + " tries remaining");
    }


}