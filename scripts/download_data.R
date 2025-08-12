library(processx)
library(httr)
#Put your data here

srrs <- readLines("C:/Users/YLS/Documents/autismMLandDL/accessionsData261.txt")

#put your output directory here
output_dir <- "C:/Users/YLS/Documents/autismMLandDL/data_261"

#if not already created, it will be

if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

for (sra_id in srrs) {
  # Query ENA API for FTP paths
  ena_api <- paste0("https://www.ebi.ac.uk/ena/portal/api/filereport?accession=", 
                    sra_id, 
                    "&result=read_run&fields=fastq_ftp&format=tsv&download=true")
  
  # Read the TSV response
  ftp_info <- read.delim(ena_api, header = TRUE, sep = "\t")
  
  # Get fastq_ftp column and split into URLs (in case of paired-end)
  fastq_links <- strsplit(as.character(ftp_info$fastq_ftp), ";")[[1]]
  
  for (link in fastq_links) {
    # Create file name
    file_name <- basename(link)
    dest_file <- file.path(output_dir, file_name)
    
    # Prepend FTP protocol
    full_url <- paste0("https://", link)
    
    # Download file
    download.file(full_url, dest_file, mode = "wb")
  }
}



