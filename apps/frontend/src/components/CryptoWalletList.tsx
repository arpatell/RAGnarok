import support from "../lib/support.json";

export type CryptoWallet = (typeof support.wallets)[number];

interface CryptoWalletListProps {
  onCopy: (wallet: CryptoWallet) => void;
}

export function CryptoWalletList({ onCopy }: CryptoWalletListProps) {
  return (
    <ul className="coffee-wallet-list" aria-label="Crypto wallet addresses">
      {support.wallets.map((wallet) => (
        <li key={wallet.id} className="coffee-wallet-item">
          <div className="wallet-network">
            <span className="wallet-icon" aria-hidden="true">{wallet.icon}</span>
            <strong>{wallet.label}</strong>
          </div>
          <img
            className="wallet-qr"
            src={`${import.meta.env.BASE_URL}crypto-wallets/${wallet.qrImage}`}
            alt={`${wallet.label} wallet address QR code`}
            width={180}
            height={180}
            loading="lazy"
          />
          <div className="coffee-address-row wallet-address-row">
            <input type="text" readOnly value={wallet.address} aria-label={`${wallet.label} address`} />
            <button type="button" onClick={() => onCopy(wallet)}>Copy</button>
          </div>
        </li>
      ))}
    </ul>
  );
}
