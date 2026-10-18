# Turning the sandbox back on, on Linux

IncarnaMind runs an Analysis in a sandbox that has no access to your files or the network. That sandbox is Chromium's, the same one the rest of the app runs in, so an Analysis can't run while the app runs without it. Instead, the place where the Analysis would have appeared says "Analyses need the app's sandbox, which is off", and the rest of the app works as usual.

## Why it is off

Some Linux distributions block the way Electron starts its sandbox. Ubuntu 24.04, for example, has an AppArmor rule that refuses unprivileged user namespaces to programs without a profile that allows them. When the sandbox can't start, the AppImage doesn't open at all, so people start it with `--no-sandbox`, which turns the sandbox off for the whole app.

## The safe fix: an AppArmor profile for the IncarnaMind AppImage's path

This allows user namespaces for the program at the IncarnaMind AppImage's path. The AppImage runs its program from a temporary folder whose name starts with `.mount_`, so the profile names that path. Any program started from a folder of that name gets the same right, which is still far narrower than the last resort below.

1. Save this as `/etc/apparmor.d/incarnamind`:

   ```
   abi <abi/4.0>,
   include <tunables/global>

   profile incarnamind /tmp/.mount_*/incarnamind flags=(unconfined) {
     userns,

     include if exists <local/incarnamind>
   }
   ```

2. Load it:

   ```sh
   sudo apparmor_parser -r /etc/apparmor.d/incarnamind
   ```

For a tighter profile, extract the AppImage to a fixed folder only an administrator can change, and name that path instead:

```sh
./IncarnaMind-*.AppImage --appimage-extract
sudo mv squashfs-root /opt/incarnamind
```

Then use `profile incarnamind /opt/incarnamind/incarnamind flags=(unconfined) {` in the profile, load it again, and start IncarnaMind with `/opt/incarnamind/AppRun`.

## The last resort: allow user namespaces for every program

```sh
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

This lifts the rule for the whole system, not just for IncarnaMind. Any program can then create user namespaces again. Attacks on the kernel often use them, which is why Ubuntu added the rule. The setting lasts until you restart the computer. To keep it after that, put `kernel.apparmor_restrict_unprivileged_userns=0` in a file in `/etc/sysctl.d/`. Use this only if the profile above doesn't work for you.

Then start IncarnaMind without `--no-sandbox`.
