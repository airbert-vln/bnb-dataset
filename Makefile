all :
	download_listings 
.PHONY : all

help :           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

lfs:
	curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
	sudo apt-get install git-lfs
	git lfs install
	git restore --source=HEAD :/

install:
	poetry install
	poetry run python -c "import nltk; nltk.download('omw-1.4')"
