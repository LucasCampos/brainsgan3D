realpath HighRes/* | sort > high_res.txt
realpath LowRes/* | sort > low_res.txt
realpath Mask/* | sort > mask.txt

paste high_res.txt mask.txt low_res.txt  -d ' ' > file_list.txt
rm high_res.txt low_res.txt mask.txt
