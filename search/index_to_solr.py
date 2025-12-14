"""
Adds the metadata documents from JSON files to Solr
The checking logic assumes that the documents are grouped in JSON files with a specific naming pattern:
solr_sound_documents_(EXPORT_NUM)_(START_ID)_(END_ID).json
"""
import json
import glob
import pysolr
import argparse
from configs import SOLR_URL, SEARCH_DOCUMENTS_DIR, BATCH_SIZE
from tqdm import tqdm

class SolrIndexer:
	def __init__(self, solr_url=SOLR_URL):
		self.solr = pysolr.Solr(solr_url, always_commit=False)

	def get_status(self):
		# Count only parent sound documents (not child similarity vectors)
		total = self.solr.search('content_type:s').hits
		if total > 0:
			min_id = self.solr.search('content_type:s', rows=1, sort='id asc').docs[0]['id']
			max_id = self.solr.search('content_type:s', rows=1, sort='id desc').docs[0]['id']
		else:
			min_id = max_id = None

		return {'total': total, 'min_id': min_id, 'max_id': max_id}
	
	def get_indexed_count_in_range(self, start_id, end_id):
		# Count only parent sound documents in this range
		query = f'content_type:s AND id:[{start_id} TO {end_id}]'
		return self.solr.search(query).hits
	
	def index_documents(self, docs, batch_size=BATCH_SIZE):
		for i in tqdm(range(0, len(docs), batch_size), desc="Indexing documents", leave=False):
			batch = docs[i:i+batch_size]
			self.solr.add(batch)
	
	def commit(self):
		self.solr.commit()

	def clear_index(self):
		print("Clearing Solr index (*:*)...")
		self.solr.delete(q='*:*')
		self.solr.commit()
		print("Index cleared.")

def get_available_files(docs_dir=SEARCH_DOCUMENTS_DIR):
	files = sorted(glob.glob(f"{docs_dir}/*.json"))

	file_info = []
	for filepath in files:
		basename = filepath.split('/')[-1]
		parts = basename.replace('.json', '').split('_')

		file_info.append({
			'path': filepath,
			'file_num': int(parts[3]),
			'start_id': int(parts[4]),
			'end_id': int(parts[5])
	})

	return file_info

def show_status(indexer, files):
	status = indexer.get_status()
	print("Solr Index Status")
	print("-"*20)
	print(f"Total indexed sounds: {status['total']}")
	if status['total'] > 0:
		print(f"ID range: {status['min_id']} - {status['max_id']}")
	
	print("\nChecking files...")
	needs_indexing = []
	for f in tqdm(files, desc="Checking"):
		count = indexer.get_indexed_count_in_range(f['start_id'], f['end_id'])
		if count < BATCH_SIZE:
			f['indexed_count'] = count
			needs_indexing.append(f)
		
	fully_indexed = len(files) - len(needs_indexing)
	print(f"\nFiles fully indexed: {fully_indexed}/{len(files)}")

	return needs_indexing

def index_files(indexer, files):
	for file_info in tqdm(files, desc="Indexing files"):
		docs = json.load(open(file_info['path']))
		# Add IDs to child documents (Solr requires all docs to have IDs) and rename for nested docs
		for doc in docs:
			if 'similarity_vectors' in doc:
				parent_id = doc['id']
				for child in doc['similarity_vectors']:
					if 'id' not in child:
						child['id'] = f"{parent_id}_{child.get('similarity_space', 'vec')}"
				
				# Rename to _childDocuments_ so Solr recognizes them as nested documents automatically
				doc['_childDocuments_'] = doc.pop('similarity_vectors')
		indexer.index_documents(docs)
	indexer.commit()

def parse_range(range_str):
	"""Parse range like '1-10' or just '10'"""
	if '-' in range_str:
		start, end = range_str.split('-')
		return int(start), int(end)
	else:
		# Just a number means 1-N
		return 1, int(range_str)

def main():
	parser = argparse.ArgumentParser(description='Index search documents into Solr')
	parser.add_argument('--status', action='store_true', help='Show current indexing status')
	parser.add_argument('--clear', action='store_true', help='Clear the index before processing')
	parser.add_argument('--index', type=str, metavar='RANGE', help='Index file range (e.g., "10" for 1-10, "5-15" for files 5-15)')
	parser.add_argument('--index-new', type=int, metavar='N', help='Index next N unindexed files')
	parser.add_argument('--index-all', action='store_true', help='Index all remaining files')
	
	args = parser.parse_args()

	indexer = SolrIndexer(SOLR_URL)
	files = get_available_files(SEARCH_DOCUMENTS_DIR)

	if args.clear:
		indexer.clear_index()

	if args.status:
		show_status(indexer, files)

	elif args.index:
		# Parse range
		start, end = parse_range(args.index)

		if start < 1 or end > len(files) or start > end:
			print(f"Invalid range. Available files: 1-{len(files)}")
			return

		# Get target files (convert to 0-indexed)
		target_files = files[start-1:end]

		needs_indexing = []

		for f in tqdm(target_files, desc="Checking files"):
			count = indexer.get_indexed_count_in_range(f['start_id'], f['end_id'])
			if count < BATCH_SIZE:
				f['indexed_count'] = count
				needs_indexing.append(f)

		if not needs_indexing:
			print(f"Files {start}-{end} already fully indexed!")
			return

		print(f"Target: files {start}-{end} ({len(target_files)} files)")
		print(f"Already indexed: {len(target_files) - len(needs_indexing)}")
		print(f"Indexing {len(needs_indexing)} files (~{len(needs_indexing)*BATCH_SIZE:,} sounds)...")

		index_files(indexer, needs_indexing)

		new_status = indexer.get_status()
		print(f"\nTotal sounds indexed: {new_status['total']:,}")

	elif args.index_new or args.index_all:
	# Find all unindexed files
		needs_indexing = []
		for f in tqdm(files, desc="Checking files"):
			count = indexer.get_indexed_count_in_range(f['start_id'], f['end_id'])
			if count < BATCH_SIZE:
				f['indexed_count'] = count
				needs_indexing.append(f)

		if not needs_indexing:
			print("All files already indexed.")
			return

		# Select how many to index
		if args.index_all:
			to_index = needs_indexing
		else:
			to_index = needs_indexing[:args.index_new]

		print(f"Found {len(needs_indexing)} unindexed files")
		print(f"Indexing {len(to_index)} files (~{len(to_index)*1000:,} sounds)...")
		
		index_files(indexer, to_index)

		new_status = indexer.get_status()
		print(f"\nTotal sounds indexed: {new_status['total']:,}")
	
	else:
		parser.print_help()

if __name__ == "__main__":
	main()