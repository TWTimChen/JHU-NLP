import argparse
import re
import sys

"""This is a validator for the sentence splitting portion of the first 605.646 Lab. It checks for:
   - presence of characters other than digits and white space
   - blank lines
   - incorrect declared number of offsets (i.e., first number on line doesn't match number of offsets provided)
   - offsets not in strictly increasing order

If a source file (second argument to the program) is provided, it also checks for:
   - submission has different number of lines than source file
   - a provided offset is past the last character of the source line
"""


def fail(linenum, message):
    if linenum >= 0:
        message = f"Line {linenum + 1}: {message}"
    sys.exit(message)

def validate_line(linenum, line, sourceline):
    if re.search(r"[^0-9 ]", line):
        fail(linenum, "File contains characters other than digits, spaces, and newlines")
    if not re.search(r"\d", line):
        fail(linenum, "File contains blank line(s)")
    nums = list(map(int, line.split()))
    num_declared_offsets = nums.pop(0)
    num_provided_offsets = len(nums)
    if num_declared_offsets != num_provided_offsets:
        fail(linenum, f"Number of offsets declared ({num_declared_offsets}) does not match number of offsets provided ({num_provided_offsets})")
    for index in range(num_provided_offsets - 1):
        if nums[index] >= nums[index + 1]:
            fail(linenum, "Offsets are not in increasing numerical order")
    if sourceline:
        if nums[-1] >= len(sourceline):
            fail(linenum, "Offset past end of line")

def validate(filename, sourcefile):
    with open(filename, "r") as infile:
        lines = infile.readlines()
    sourcelines = None
    if sourcefile:
        with open(sourcefile, "r") as infile:
            sourcelines = infile.readlines()
        if len(sourcelines) != len(lines):
            fail(-1, f"Number of lines in submission file ({len(lines)}) does not match number of lines in source file ({len(sourcelines)})")
    for linenum, line in enumerate(lines):
        validate_line(linenum, line.rstrip('\n'), sourcelines[linenum].rstrip('\n') if sourcefile else None)
    print("No problems found.")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description = "Validate a 605.646 Lab 1 Part D output file")
    parser.add_argument('submissionfile', help="File containing output of your Lab 1 Part D (sentence boundary detection) program")
    parser.add_argument('sourcefile', nargs='?', help="Optional source file of lines that were split into sentences")

    args = vars(parser.parse_args())
    validate(args['submissionfile'], args['sourcefile'])

if __name__ == '__main__':
    main()