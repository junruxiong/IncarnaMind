import Link from 'next/link';
import getUserSession from '@/lib/getUserSession';
import createSupabaseServerClient from '@/lib/supabase/server';

const Header = async () => {
  const { data } = await getUserSession();

  const logoutAction = async () => {
    'use server';
    const supabase = await createSupabaseServerClient();
    await supabase.auth.signOut();
  };

  return (
    <header className='bg-white h-20'>
      <nav className='h-full flex justify-between container items-center'>
        <div>
          <Link href='/' className='text-ct-dark-600 text-2xl font-semibold'>
            CodevoWeb
          </Link>
        </div>
        <ul className='flex items-center space-x-4'>
          <li>
            <Link href='/' className='text-ct-dark-600'>
              Home
            </Link>
          </li>
          {!data.session && (
            <>
              <li>
                <Link href='/register' className='text-ct-dark-600'>
                  Register
                </Link>
              </li>
              <li>
                <Link href='/login' className='text-ct-dark-600'>
                  Login
                </Link>
              </li>
            </>
          )}
          {data.session && (
            <form action={logoutAction} className='flex'>
              <li>
                <Link href='/profile' className='text-ct-dark-600'>
                  Profile
                </Link>
              </li>
              <li>
                <button className='ml-4'>Logout</button>
              </li>
            </form>
          )}
        </ul>
      </nav>
    </header>
  );
};

export default Header;
