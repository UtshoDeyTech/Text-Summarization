@echo off

REM Check if virtual environment exists, create if not
IF NOT EXIST env\Scripts\activate (
    echo Creating virtual environment...
    python -m venv env
) ELSE (
    echo Virtual environment already exists.
)

REM Activate virtual environment
call .\env\Scripts\activate

REM Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create .env file if it doesn't exist and add OPENAI_API_KEY line
IF NOT EXIST .env (
    echo Creating .env file...
    echo OPENAI_API_KEY= > .env
) ELSE (
    echo .env file already exists.
)

pause
