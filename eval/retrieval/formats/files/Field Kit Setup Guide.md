# Field Kit Setup Guide

This guide is for survey teams taking the standard field kit out for more than one night. Read it before your first trip and keep a printed copy in Bag A.

## Packing list

| Item | Weight | Packed in | Notes |
|---|---|---|---|
| Tent (two-person) | 2.1 kg | Bag A | Pegs in the side pocket |
| Water filter | 0.4 kg | Bag B | Replace the cartridge after 1,000 litres |
| Satellite messenger | 0.15 kg | Chest pouch | Charge to 100% the night before |
| Stove and fuel | 0.9 kg | Bag B | Fuel canisters can't fly in hold luggage |
| First aid kit | 0.6 kg | Bag A | Restock after every trip |

Bag A should weigh under 9 kg when packed; Bag B under 7 kg.

## Before you leave

1. Register the trip plan with the base office at least 48 hours ahead.
2. Test the satellite messenger by sending a check-in on channel 7.
3. Photograph the vehicle's fuel gauge and mileage.
4. Leave the spare vehicle key in the key safe at the depot.

Things to check on the weather forecast:

- Wind above 50 km/h on exposed ridges means the trip is postponed.
- Thunderstorms within 24 hours mean no camping above the tree line.
- A river level above 1.2 m at the Hollin gauge closes the ford.

## Syncing data

The kit's tablet syncs survey records to the base server whenever it has a signal. Its settings live in `sync.ini`:

```ini
[sync]
server = sync.fieldkit.example
port = 8443
interval_minutes = 20
retry_limit = 5
compress = true
```

If a sync fails five times in a row, the tablet stores records locally until the next manual sync.

Survey records are exported as CSV with one row per observation. The first line of every export is the header:

```csv
site_id,observed_at,observer,species_code,count,notes
```

Times are in UTC, written as ISO 8601. Species codes follow the regional four-letter list, so a curlew is CURL and a lapwing is LAPW.

## Charging and power

The kit runs on two power banks and a folding solar panel. In cloud the panel gives about a third of its rated output, so plan on the power banks alone for overcast trips.

| Device | Battery | Lasts | Charge from |
|---|---|---|---|
| Survey tablet | 7,600 mAh | 9 hours of fieldwork | Power bank 1 |
| Satellite messenger | 2,000 mAh | 4 days with 10-minute tracking | Power bank 2 |
| Head torch | 1,200 mAh | 6 hours on the medium setting | Either bank |

Keep the power banks inside your sleeping bag on cold nights: below freezing they can lose half their charge overnight.

## Radio procedure

Call the base office at 08:00 and 18:00 every day you are out, even when there is nothing to report. If a call is missed, base tries again after thirty minutes and then raises the alarm after a second missed call. Use plain language and give your grid reference to the nearest hundred metres.

## In an emergency

- Press SOS on the satellite messenger and keep it switched on.
- Stay with the vehicle or the tent unless it is unsafe to do so.
- Give first aid, then send a message with the number of people injured.

## After the trip

Dry the tent before it goes back into storage, and log any broken equipment in the kit register within two days. Return the satellite messenger to the charging shelf in the store room, not to a desk drawer, so that the next team finds it charged.
