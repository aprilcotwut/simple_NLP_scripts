# CSCE 5543-Statistical NLP Homework 1 
## Due Date - Fri., Sept. 20, 9:40am 
Students may work in pairs
Problems from the textbook
Each individual must first answer each of the following questions. After all questions are complete, you may get together with a partner to review/improve your individual solutions. The pair should then submit a single, joint solution with the original, individual solutions attached. If you cannot agree on an answer, you may submit individual solutions to particular questions as necessary Do the following questions from Manning and Shutze Section 2.1.11 (Exercises):
2.1 (a)
2.3
2.8
Programming Problem
The goal is to 1) identify and count word pairs and 2) to identify word collocations. You will first create one table of bigrams in decreasing order by their frequency of occurrence. You will first create a second table of bigrams that are ranked using a method to identify those that are true collocations (i.e., have a separate meaning from that indicated by the contributing words). You will then manually judge the top 100 bigrams in each set as to whether or not they are actually collocations (phrases whose meaning is not immediately obvious from the two words), and report on the accuracy of both methods.

The files to use for this assignment can be copied from sgauch/public_html/5543/amazon_reviews.txt on turing or downloaded from amazon_reviews.txt.
For developing your code, begin working with a subset of this file (which has one review per line) rather than the whole thing because you may run into issues with the size of the vocabulary that need addressing. If you are unable to use the entire file, please mention that in your final report and document how big a vocabulary you can work with.

*2. Design:*
Create or modify a program to count the frequencies of all word pairs rather than single words. Be sure to look for word pairs after all preprocessing is done. You may choose to have a hash-table entry for each word pair or create a hashtable with the first word as the key and a list or other storage structure for all second words seen after the first (key) word. Note, you need to use some sort of sparse representation, not a full 2-d array.

You may assume that all the words and frequency information will fit in memory. You do not need to store intermediate information on disk, although you may choose to do so.

*3. Implementation:*
You may use C++ or Java or Python on any development platform. Your entire program (if it has more than one step) should run from the command line with a single call to an executable or a batch/shell script. You may use any tokenizer you like, but here are some sample single token tokenizers that you may choose to use and/or modify (they were teseted last year, but may not work this year... but they should be close!):

README for the tokenizers
C++ text processor
Java text processor
Python text processor
NLTK + Python text processor
You may choose to use this tokenizer or write your own or borrow one from anywhere (with attribution). NLTK could be very useful for this. You may write a single program that tokenizes and counts, or create intermediate (tokenized) intermediate files and then run your bigram counter on those files. If you choose the latter approach, you need to create a shell script so that, regardless of your approach, your program can be executed simply, e.g.,

count_bigrams input-directory output-directory
Your program should count the frequency information for all bigrams. To keep this scalable, you may eliminate from consideration any tokens that appear only once in the collection. It should then sort the output and display the top 25 most frequent bigrams to the screen, in decreasing order by frequency.

Now implement some method to try to locate true collocations. You may use any method in the book (mutual information, chi-squared, maximum likelihood) or invent your own. This can run as a different program or your program can do both methods (frequency based and intersting) at the same time. It may make use of files created by the count_bigrams program, but it must run from a single command, e.g.,

find_collocations input-file_or_directory output-file_or_directory
*4. Testing:*
Test your program to check that it operates correctly for all of the requirements listed above. Also check for the error handling capabilities of the code. Save your testing output in text files for submission on the program due date.

*5. Documentation:*
When you have completed your program, write a short report (approximately two pages long) describing your algorithms, data structures and discussing the computational complexity (N, N-squared?) of your solutions with respect to space and time (where N is the number of tokens and/or unique words in the corpus). Also, include information about how long your program takes to run for each method.

Discuss your text preprocessing and what decisions you made on handling punctuation and other text normalization.

Finally, discuss the resulting top and bottom weighted bigrams for each method. For each, calculate the percent of the top 100 that you think are true collocations (i.e., 10/100 = 10%). Attach the top 100 highest weighted and the bottom 5 lowest weighted bigrams for each method to the report.

*6. Project Submission:*
Turn your printed report in to class at the beginning of the session or in the CSCE main office. If you work in pairs on either or both parts, only turn in one report for the parts done together with both names on it. Late assignments will be deducted 10% per 24 hour period. No late assignments will be accepted after 3 days. Weekends (Friday to Monday) count as one day.
