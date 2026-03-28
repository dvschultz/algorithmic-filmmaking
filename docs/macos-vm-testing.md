# Testing with a macOS VM (UTM)

Guide for testing Scene Ripper on a clean macOS installation using UTM.

## Why UTM

- Free and open-source
- Uses Apple's Virtualization.framework (native performance on Apple Silicon)
- Simple GUI, snapshots, easy to reset to clean state
- Alternatives: Tart (CLI-based, free), VMware Fusion (free personal license), Parallels ($100/yr)

## Setup

### 1. Install UTM

```bash
brew install --cask utm
```

Or download from [utm.app](https://mac.getutm.app).

### 2. Create a macOS VM

Open UTM > Click **+** (Create New) > **Virtualize** > **macOS**

UTM will automatically fetch the latest available IPSW. If you want a specific macOS version, you can download IPSWs from Apple directly -- UTM lets you browse or provide your own.

### 3. Configure the VM

UTM walks you through a wizard:

| Setting | Recommendation |
|---------|---------------|
| RAM | At least 8 GB (half your host RAM is a safe max) |
| CPU cores | 4+ |
| Disk | 64 GB minimum, 80-100 GB comfortable |
| Display | Default is fine |

### 4. Install macOS

Click Play. macOS installs automatically (15-30 minutes). Go through the standard setup assistant. Skip Apple ID if you want a clean test environment.

### 5. Snapshot the clean state

Once setup is done and you're at the desktop:

1. Shut down the VM
2. In UTM, right-click the VM > **Clone**

This gives you a pristine baseline to restore after each test.

## Testing the App

Inside the VM:

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Test from DMG (CI build)
# Open the .dmg file, drag to Applications, launch

# Or test from source
pip install -r requirements-core.txt
python main.py
```

## Tips

- **Shared folder**: UTM supports shared directories between host and guest -- drag your DMG or repo in without needing to re-download
- **No Apple ID needed**: Skip sign-in for a clean test environment
- **Network works out of the box**: The VM gets NAT networking by default
- **Reset between tests**: Shut down > delete the VM > re-clone from your clean snapshot
